# EVOLVE-BLOCK-START
import sys
import os
import json
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using layer-based optimization.
    
    For layered DAGs, process layer by layer with smart memory management
    to minimize total load/store transfers.
    """
    try:
        vertices, parents, inputs, outputs, topo_order = _parse_dag(dag)
    except Exception:
        return _fallback_naive(dag, fast_memory_budget)
    
    if vertices is None or len(vertices) == 0:
        return []
    
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    levels = _compute_levels(vertices, parents, topo_order)
    max_level = max(levels.values()) if levels else 0
    
    level_groups = defaultdict(list)
    for v in vertices:
        level_groups[levels[v]].append(v)
    
    fast_mem = set(inputs)
    slow_mem = set()
    ops = []
    
    for lvl in range(max_level + 1):
        verts_at_level = level_groups.get(lvl, [])
        if not verts_at_level:
            continue
        
        for v in verts_at_level:
            if v in inputs:
                continue
            
            ps = parents.get(v, [])
            for p in ps:
                if p not in fast_mem:
                    if p in slow_mem:
                        if len(fast_mem) >= fast_memory_budget:
                            evicted = _evict_smart(fast_mem, slow_mem, levels, outputs, v, p, children, lvl)
                            if evicted is not None:
                                fast_mem.discard(evicted)
                                ops.append(["delete", evicted])
                        ops.append(["load", p])
                        fast_mem.add(p)
            
            if len(fast_mem) >= fast_memory_budget:
                evicted = _evict_smart(fast_mem, slow_mem, levels, outputs, v, None, children, lvl)
                if evicted is not None:
                    fast_mem.discard(evicted)
                    ops.append(["delete", evicted])
            
            ops.append(["compute", v])
            fast_mem.add(v)
        
        next_level_verts = set(level_groups.get(lvl + 1, []))
        needed_for_next = set()
        for v in verts_at_level:
            for c in children.get(v, []):
                if levels.get(c, -1) == lvl + 1:
                    needed_for_next.add(v)
                    break
        
        output_set = set(outputs)
        to_keep = (needed_for_next | output_set) & fast_mem
        
        if len(to_keep) > fast_memory_budget:
            priority = []
            for v in to_keep:
                is_needed = v in needed_for_next
                is_output = v in output_set
                score = (0 if is_needed else 1, 0 if is_output else 1, levels.get(v, 0))
                priority.append((score, v))
            priority.sort()
            to_keep = set(v for _, v in priority[:fast_memory_budget])
        
        for v in verts_at_level:
            if v in inputs:
                continue
            if v in fast_mem:
                if v in to_keep:
                    pass
                else:
                    if v in needed_for_next or v in output_set:
                        if v not in slow_mem:
                            slow_mem.add(v)
                            ops.append(["store", v])
                        fast_mem.discard(v)
                    else:
                        ops.append(["delete", v])
                        fast_mem.discard(v)
                        if v in slow_mem:
                            slow_mem.discard(v)
    
    for o in outputs:
        if o not in fast_mem:
            if o in slow_mem:
                if len(fast_mem) >= fast_memory_budget:
                    evicted = _evict_smart(fast_mem, slow_mem, levels, outputs, o, None, children, max_level)
                    if evicted is not None:
                        fast_mem.discard(evicted)
                        ops.append(["delete", evicted])
                ops.append(["load", o])
                fast_mem.add(o)
            else:
                _recompute(o, fast_mem, slow_mem, parents, children, levels, outputs, fast_memory_budget, ops)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops


def _parse_dag(dag):
    if isinstance(dag, dict):
        if 'vertices' in dag:
            vertices = list(dag['vertices'])
            parents = {v: list(dag['parents'].get(v, [])) for v in vertices}
            inputs = set(dag.get('inputs', []))
            outputs = set(dag.get('outputs', []))
            topo_order = _toposort(vertices, parents)
            return vertices, parents, inputs, outputs, topo_order
        
        first_val = next(iter(dag.values())) if dag else None
        if isinstance(first_val, list):
            vertices = list(dag.keys())
            parents = {k: list(v) for k, v in dag.items()}
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            topo_order = _toposort(vertices, parents)
            return vertices, parents, inputs, outputs, topo_order
        
        if 'nodes' in dag and 'edges' in dag:
            nodes = dag['nodes']
            edges = dag['edges']
            vertices = list(nodes) if isinstance(nodes, list) else list(nodes.keys())
            parents = {v: [] for v in vertices}
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    src, dst = e[0], e[1]
                    if dst in parents:
                        parents[dst].append(src)
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            topo_order = _toposort(vertices, parents)
            return vertices, parents, inputs, outputs, topo_order
        
        if 'n' in dag:
            n = dag['n']
            vertices = list(range(n))
            parents_raw = dag.get('parents', [])
            parents = {}
            for i in range(n):
                parents[i] = list(parents_raw[i]) if i < len(parents_raw) else []
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            topo_order = _toposort(vertices, parents)
            return vertices, parents, inputs, outputs, topo_order
    
    elif isinstance(dag, (list, tuple)):
        if len(dag) > 0 and isinstance(dag[0], (list, tuple)) and len(dag[0]) == 2:
            vertices = []
            parents = {}
            for item in dag:
                v, ps = item[0], item[1]
                vertices.append(v)
                parents[v] = list(ps)
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            topo_order = _toposort(vertices, parents)
            return vertices, parents, inputs, outputs, topo_order
    
    return None, None, None, None, None


def _toposort(vertices, parents):
    in_degree = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
                in_degree[v] += 1
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    if len(order) < len(vertices):
        for v in vertices:
            if v not in order:
                order.append(v)
    return order


def _compute_levels(vertices, parents, topo_order):
    levels = {}
    for v in topo_order:
        ps = parents.get(v, [])
        if not ps:
            levels[v] = 0
        else:
            levels[v] = max(levels.get(p, 0) for p in ps) + 1
    return levels


def _evict_smart(fast_mem, slow_mem, levels, outputs, current, priority, children, current_level):
    candidates = []
    for v in fast_mem:
        if v == current or v == priority or v in outputs:
            continue
        level = levels.get(v, 0)
        candidates.append((level, v))
    if not candidates:
        for v in fast_mem:
            if v != current and v != priority and v not in outputs:
                return v
        return None
    candidates.sort()
    return candidates[0][1]


def _recompute(v, fast_mem, slow_mem, parents, children, levels, outputs, budget, ops):
    if v in fast_mem:
        return
    if v in slow_mem:
        if len(fast_mem) >= budget:
            evicted = _evict_smart(fast_mem, slow_mem, levels, outputs, v, None, children, levels.get(v, 0))
            if evicted is not None:
                fast_mem.discard(evicted)
                ops.append(["delete", evicted])
        ops.append(["load", v])
        fast_mem.add(v)
        return
    for p in parents.get(v, []):
        _recompute(p, fast_mem, slow_mem, parents, children, levels, outputs, budget, ops)
    if len(fast_mem) >= budget:
        evicted = _evict_smart(fast_mem, slow_mem, levels, outputs, v, None, children, levels.get(v, 0))
        if evicted is not None:
            fast_mem.discard(evicted)
            ops.append(["delete", evicted])
    ops.append(["compute", v])
    fast_mem.add(v)


def _fallback_naive(dag, fast_memory_budget):
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    except Exception:
        return []

# EVOLVE-BLOCK-END
