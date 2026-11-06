# EVOLVE-BLOCK-START
import sys
import os
import json
import time
from collections import defaultdict, deque

def _get_module():
    """Import the task module to inspect DAG structure and baseline."""
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        import autoresearch_suite.science_tcs_expansion as mod
        return mod
    except Exception:
        return None

def _safe_baseline(dag, fast_memory_budget):
    """Run baseline, return its operations list."""
    mod = _get_module()
    if mod is not None and hasattr(mod, 'pebble_baseline'):
        try:
            return mod.pebble_baseline(dag, fast_memory_budget)
        except Exception as e:
            return None
    return None

def _parse_dag(dag):
    """
    Parse dag into (vertices, parents, inputs, outputs, layer_of).
    """
    if isinstance(dag, dict):
        if 'vertices' in dag:
            vertices = dag['vertices']
            parents = dag.get('parents', {})
            inputs = dag.get('inputs', set())
            outputs = dag.get('outputs', set())
            if isinstance(inputs, list):
                inputs = set(inputs)
            if isinstance(outputs, list):
                outputs = set(outputs)
            layer_of = dag.get('layer', {})
            return vertices, parents, inputs, outputs, layer_of
        
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
            return vertices, parents, inputs, outputs, {}
        
        if 'nodes' in dag and 'edges' in dag:
            nodes = dag['nodes']
            edges = dag['edges']
            vertices = nodes if isinstance(nodes, list) else list(nodes.keys())
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
            return vertices, parents, inputs, outputs, {}
        
        if 'n' in dag:
            n = dag['n']
            vertices = list(range(n))
            parents_raw = dag.get('parents', [])
            parents = {}
            for i in range(n):
                if i < len(parents_raw):
                    parents[i] = list(parents_raw[i])
                else:
                    parents[i] = []
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs, {}
    
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
            return vertices, parents, inputs, outputs, {}
    
    mod = _get_module()
    if mod is not None:
        for attr in ['parse_dag', 'get_dag_info', 'extract_dag']:
            if hasattr(mod, attr):
                try:
                    result = getattr(mod, attr)(dag)
                    if isinstance(result, tuple) and len(result) >= 4:
                        return result + ({},) * (5 - len(result))
                except Exception:
                    pass
    
    return None, None, None, None, {}

def _compute_layers(vertices, parents, inputs, outputs):
    """Compute layer assignment (longest path from inputs)."""
    layer_of = {}
    for v in vertices:
        if v in inputs:
            layer_of[v] = 0
        else:
            ps = parents.get(v, [])
            if ps:
                layer_of[v] = max(layer_of.get(p, 0) + 1 for p in ps)
            else:
                layer_of[v] = 0
    return layer_of

def _evict_lookahead(fast_mem, children, outputs, protected, layer_of):
    """Choose eviction candidate based on lookahead."""
    candidates = []
    for v in fast_mem:
        if v in protected or v in outputs:
            continue
        ch = children.get(v, [])
        uncomputed = sum(1 for c in ch if layer_of.get(c, 0) > layer_of.get(v, 0))
        candidates.append((uncomputed, layer_of.get(v, 0), v))
    if not candidates:
        for v in fast_mem:
            if v not in protected and v not in outputs:
                return v
        return None
    candidates.sort()
    return candidates[0][2]

def _ensure_computed(v, fast_mem, computed, parents, children, outputs, budget, ops, layer_of):
    """Ensure vertex v is in fast memory."""
    if v in fast_mem:
        return
    ps = parents.get(v, [])
    for p in ps:
        _ensure_computed(p, fast_mem, computed, parents, children, outputs, budget, ops, layer_of)
    if len(fast_mem) >= budget:
        evicted = _evict_lookahead(fast_mem, children, outputs, {v}, layer_of)
        if evicted is not None:
            fast_mem.discard(evicted)
            ops.append(["delete", evicted])
    ops.append(["compute", v])
    fast_mem.add(v)
    computed.add(v)

def _pebble_layered(vertices, parents, inputs, outputs, budget, layer_of):
    """Layer-by-layer pebbling with lookahead eviction."""
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    max_layer = max(layer_of.values()) if layer_of else 0
    layers = defaultdict(list)
    for v in vertices:
        layers[layer_of[v]].append(v)
    for l in layers:
        layers[l].sort()
    
    fast_mem = set(inputs)
    computed = set(inputs)
    ops = []
    
    for l in range(1, max_layer + 1):
        current_layer = layers.get(l, [])
        if not current_layer:
            continue
        
        needed_parents = set()
        for v in current_layer:
            for p in parents.get(v, []):
                needed_parents.add(p)
        
        for p in sorted(needed_parents):
            if p not in fast_mem:
                if len(fast_mem) >= budget:
                    evicted = _evict_lookahead(fast_mem, children, outputs, needed_parents | set(current_layer), layer_of)
                    if evicted is not None:
                        fast_mem.discard(evicted)
                        ops.append(["delete", evicted])
                _ensure_computed(p, fast_mem, computed, parents, children, outputs, budget, ops, layer_of)
        
        for v in current_layer:
            if len(fast_mem) >= budget:
                evicted = _evict_lookahead(fast_mem, children, outputs, set(current_layer) | needed_parents, layer_of)
                if evicted is not None:
                    fast_mem.discard(evicted)
                    ops.append(["delete", evicted])
            ops.append(["compute", v])
            fast_mem.add(v)
            computed.add(v)
        
        for v in list(fast_mem):
            if v not in outputs and layer_of.get(v, 0) < l:
                ch = children.get(v, [])
                if all(c in computed for c in ch):
                    fast_mem.discard(v)
                    ops.append(["delete", v])
    
    for o in sorted(outputs):
        if o not in fast_mem:
            if len(fast_mem) >= budget:
                evicted = _evict_lookahead(fast_mem, children, outputs, {o}, layer_of)
                if evicted is not None:
                    fast_mem.discard(evicted)
                    ops.append(["delete", evicted])
            _ensure_computed(o, fast_mem, computed, parents, children, outputs, budget, ops, layer_of)
    
    return ops

def pebble(dag, fast_memory_budget):
    """Communication-minimizing DAG pebbling."""
    baseline_ops = _safe_baseline(dag, fast_memory_budget)
    
    try:
        vertices, parents, inputs, outputs, layer_of = _parse_dag(dag)
    except Exception:
        return baseline_ops if baseline_ops is not None else []
    
    if vertices is None:
        return baseline_ops if baseline_ops is not None else []
    
    if not layer_of:
        layer_of = _compute_layers(vertices, parents, inputs, outputs)
    
    ops = _pebble_layered(vertices, parents, inputs, outputs, fast_memory_budget, layer_of)
    
    if not ops:
        return baseline_ops if baseline_ops is not None else []
    
    if len(ops) > 20000:
        return baseline_ops if baseline_ops is not None else ops[:20000]
    
    return ops

# EVOLVE-BLOCK-END
