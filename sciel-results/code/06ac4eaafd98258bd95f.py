# EVOLVE-BLOCK-START
import sys
import os
import json
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Improved Strategy:
    1. Parse DAG.
    2. Greedy topological processing.
    3. Smart Eviction:
       - Prefer evicting vertices that are in slow memory (safe to evict, can reload).
       - Prefer evicting vertices with 0 remaining uses (safe to evict, won't be needed).
       - Avoid evicting vertices not in slow memory with >0 remaining uses (forces recompute).
    4. If forced to recompute, do so recursively but carefully.
    """
    try:
        vertices, parents, inputs, outputs = _parse_dag(dag)
    except Exception:
        return []
    
    if vertices is None:
        return []
        
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
            
    # Topological order is assumed to be the order of vertices list
    topo_order = list(vertices)
    
    # Precompute remaining uses
    remaining_uses = {v: len(children[v]) for v in vertices}
    
    # State
    fast_mem = set(inputs)
    slow_mem = set(inputs) # Inputs are usually "available" or we assume they are loaded initially? 
                           # "Inputs start blue" -> in fast memory. 
                           # Do they also exist in slow? Usually inputs are external or pre-loaded.
                           # Let's assume inputs are in fast_mem. If we evict them, can we reload?
                           # If inputs are not in slow_mem, evicting them is dangerous.
                           # Let's assume inputs are effectively in slow_mem (source of truth) or we shouldn't evict them if possible.
                           # Actually, if inputs are "blue" initially, they are in fast.
                           # If we delete them, can we load them? 
                           # "load" implies from slow to fast. If input is not in slow, we can't load.
                           # So, treat inputs as "always available" or "in slow".
    
    ops = []
    
    # Process vertices in topo order
    for v in topo_order:
        if v in inputs:
            continue
            
        ps = parents.get(v, [])
        
        # Ensure all parents are in fast memory
        for p in ps:
            if p not in fast_mem:
                if p in slow_mem:
                    # Need to load
                    if len(fast_mem) >= fast_memory_budget:
                        evicted = _evict_smart(fast_mem, slow_mem, remaining_uses, outputs, v, p, children)
                        if evicted is not None:
                            fast_mem.discard(evicted)
                            ops.append(["delete", evicted])
                    ops.append(["load", p])
                    fast_mem.add(p)
                else:
                    # Not in slow, must recompute
                    _recompute(p, fast_mem, slow_mem, parents, children, remaining_uses, outputs, fast_memory_budget, ops)
        
        # Compute v
        if len(fast_mem) >= fast_memory_budget:
            evicted = _evict_smart(fast_mem, slow_mem, remaining_uses, outputs, v, None, children)
            if evicted is not None:
                fast_mem.discard(evicted)
                ops.append(["delete", evicted])
        
        ops.append(["compute", v])
        fast_mem.add(v)
        
        # Update remaining uses for parents
        for p in ps:
            if p in remaining_uses:
                remaining_uses[p] -= 1
                
        # Store v in slow memory? 
        # "store" operation puts from fast to slow.
        # Should we store every computed value?
        # If we store it, we can reload it later instead of recomputing.
        # Cost of store: 1 transfer.
        # Cost of recompute: 1 compute + loads for parents.
        # Generally, storing is better if the vertex is used multiple times.
        # If used only once, storing is waste (unless we need to evict it from fast to make space, but then we lose it).
        # Let's store if remaining_uses[v] > 0.
        if remaining_uses.get(v, 0) > 0:
            ops.append(["store", v])
            slow_mem.add(v)
            
    # Ensure outputs are in fast memory
    for o in outputs:
        if o not in fast_mem:
            if o in slow_mem:
                if len(fast_mem) >= fast_memory_budget:
                    evicted = _evict_smart(fast_mem, slow_mem, remaining_uses, outputs, o, None, children)
                    if evicted is not None:
                        fast_mem.discard(evicted)
                        ops.append(["delete", evicted])
                ops.append(["load", o])
                fast_mem.add(o)
            else:
                _recompute(o, fast_mem, slow_mem, parents, children, remaining_uses, outputs, fast_memory_budget, ops)
                
    if len(ops) > 20000:
        return ops[:20000]
        
    return ops

def _evict_smart(fast_mem, slow_mem, remaining_uses, outputs, current_vertex, priority_parent, children):
    """
    Choose a vertex to evict.
    Priority:
    1. Not current, not priority parent, not output.
    2. In slow_mem (safe to evict).
    3. remaining_uses == 0 (won't be needed).
    4. Lowest remaining_uses.
    """
    candidates = []
    for v in fast_mem:
        if v == current_vertex or v == priority_parent:
            continue
        if v in outputs:
            continue
            
        in_slow = 1 if v in slow_mem else 0
        uses = remaining_uses.get(v, 0)
        
        # We want to minimize "pain".
        # Pain = 0 if in_slow or uses==0.
        # Pain = high if not in_slow and uses>0.
        
        # Score: lower is better.
        # If in_slow, score = uses (prefer evicting those with fewer future uses, though it's free to reload)
        # If not in_slow, score = 1000 + uses (penalize heavily)
        
        if in_slow:
            score = uses
        else:
            score = 1000 + uses
            
        candidates.append((score, v))
        
    if not candidates:
        return None
        
    candidates.sort()
    return candidates[0][1]

def _recompute(v, fast_mem, slow_mem, parents, children, remaining_uses, outputs, fast_memory_budget, ops):
    """
    Recursively recompute v if it's not in fast or slow.
    """
    if v in fast_mem:
        return
        
    if v in slow_mem:
        # Load it
        if len(fast_mem) >= fast_memory_budget:
            evicted = _evict_smart(fast_mem, slow_mem, remaining_uses, outputs, v, None, children)
            if evicted is not None:
                fast_mem.discard(evicted)
                ops.append(["delete", evicted])
        ops.append(["load", v])
        fast_mem.add(v)
        return
        
    # Recompute from parents
    ps = parents.get(v, [])
    for p in ps:
        _recompute(p, fast_mem, slow_mem, parents, children, remaining_uses, outputs, fast_memory_budget, ops)
        
    # Now compute v
    if len(fast_mem) >= fast_memory_budget:
        evicted = _evict_smart(fast_mem, slow_mem, remaining_uses, outputs, v, None, children)
        if evicted is not None:
            fast_mem.discard(evicted)
            ops.append(["delete", evicted])
            
    ops.append(["compute", v])
    fast_mem.add(v)
    
    # Store it?
    if remaining_uses.get(v, 0) > 0:
        ops.append(["store", v])
        slow_mem.add(v)

def _parse_dag(dag):
    """
    Parse dag into (vertices, parents, inputs, outputs).
    """
    if isinstance(dag, dict):
        if 'vertices' in dag:
            vertices = dag['vertices']
            parents = dag.get('parents', {})
            inputs = set(dag.get('inputs', []))
            outputs = set(dag.get('outputs', []))
            return vertices, parents, inputs, outputs
        
        if 'n' in dag:
            n = dag['n']
            vertices = list(range(n))
            parents_raw = dag.get('parents', [])
            parents = {i: list(parents_raw[i]) if i < len(parents_raw) else [] for i in range(n)}
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
            
        # Dict of vertex -> parents
        if dag:
            first_val = next(iter(dag.values()))
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
                return vertices, parents, inputs, outputs
                
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
            return vertices, parents, inputs, outputs
            
    return None, None, None, None

# EVOLVE-BLOCK-END
