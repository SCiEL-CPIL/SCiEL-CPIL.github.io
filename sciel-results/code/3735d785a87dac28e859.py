"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

def _pebble_optimized(dag, budget, deadline=None):
    """
    Optimized pebbling with need-aware eviction and lookahead.
    """
    if deadline is None:
        deadline = time.time() + 50
    
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    
    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}
    
    # Precompute recompute cost C_v
    C = [0] * n
    def calc_cost(v):
        i = idx[v]
        if C[i] != 0:
            return C[i]
        ps = parents.get(v, [])
        if not ps:
            C[i] = 1
        else:
            s = 0
            for p in ps:
                s += calc_cost(p)
            C[i] = 1 + s
        return C[i]
    
    for v in vertices:
        calc_cost(v)
        
    # Topological sort
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
            
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
                
    if len(topo) != n:
        topo = vertices
        
    outset = set(outputs)
    inputs = [v for v in vertices if not parents.get(v, [])]
    
    # Compute "need" score: how important is each vertex for reaching outputs?
    # need[v] = max(need of children that are on path to outputs)
    # For outputs, need = infinity (we must have them)
    # For others, need = sum of needs of children that depend on them
    need = [0.0] * n
    for v in reversed(topo):
        i = idx[v]
        if v in outset:
            need[i] = float('inf')
        else:
            # Sum of needs of children
            s = 0.0
            for w in children.get(v, []):
                j = idx[w]
                if need[j] > 0:
                    s += need[j]
            need[i] = s
    
    # State
    in_memory = set()
    on_disk = set()
    ops = []
    
    # Initialize memory: prioritize inputs with highest need
    if len(inputs) <= budget:
        in_memory = set(inputs)
    else:
        # Sort inputs by need (descending) and take top budget
        sorted_inputs = sorted(inputs, key=lambda x: -need[idx[x]])
        in_memory = set(sorted_inputs[:budget])
        
    for v in inputs:
        if v not in in_memory:
            on_disk.add(v)
            
    max_ops = 20000
    op_count = 0
    
    while op_count < max_ops:
        if time.time() > deadline - 2:
            break
            
        if all(o in in_memory for o in outputs):
            break
            
        # Find ready to compute
        ready_to_compute = []
        for v in vertices:
            if v in in_memory:
                continue
            ps = parents.get(v, [])
            if not ps:
                continue
            if all(p in in_memory for p in ps):
                ready_to_compute.append(v)
        
        if ready_to_compute:
            # Pick vertex with highest need (or C if need is 0)
            v_to_compute = max(ready_to_compute, key=lambda x: (need[idx[x]], C[idx[x]]))
            
            if len(in_memory) >= budget:
                # Evict: minimize (future_recompute_cost + penalty_if_output)
                best_evict = None
                best_loss = float('inf')
                
                for u in in_memory:
                    c_u = C[idx[u]]
                    need_u = need[idx[u]]
                    
                    # Cost to recover: min(store+load=2, delete+recompute=C_u)
                    recover_cost = min(2, c_u)
                    
                    # Penalty for evicting needed vertices
                    penalty = 0
                    if u in outset:
                        penalty = 10000
                    elif need_u > 0:
                        # Scale penalty by need
                        penalty = min(need_u * 10, 1000)
                    
                    loss = recover_cost + penalty
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_evict = u
                        evict_op = "store" if c_u > 2 else "delete"
                        
                if best_evict is not None:
                    if evict_op == "store":
                        ops.append(["store", best_evict])
                        on_disk.add(best_evict)
                    else:
                        ops.append(["delete", best_evict])
                    in_memory.remove(best_evict)
                    op_count += 1
                    continue
                    
            ops.append(["compute", v_to_compute])
            in_memory.add(v_to_compute)
            if v_to_compute in on_disk:
                on_disk.remove(v_to_compute)
            op_count += 1
            
        else:
            # Need to load a parent
            candidates = set()
            for v in vertices:
                if v in in_memory:
                    continue
                ps = parents.get(v, [])
                if not ps:
                    continue
                missing = [p for p in ps if p not in in_memory]
                if len(missing) == 1:
                    candidates.add(missing[0])
            
            if not candidates:
                # Load any input not in memory with highest need
                for v in inputs:
                    if v not in in_memory:
                        candidates.add(v)
                        break
            
            if not candidates:
                break
                
            v_to_load = max(candidates, key=lambda x: (need[idx[x]], C[idx[x]]))
            
            if len(in_memory) >= budget:
                best_evict = None
                best_loss = float('inf')
                
                for u in in_memory:
                    c_u = C[idx[u]]
                    need_u = need[idx[u]]
                    recover_cost = min(2, c_u)
                    penalty = 0
                    if u in outset:
                        penalty = 10000
                    elif need_u > 0:
                        penalty = min(need_u * 10, 1000)
                    loss = recover_cost + penalty
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_evict = u
                        evict_op = "store" if c_u > 2 else "delete"
                        
                if best_evict is not None:
                    if evict_op == "store":
                        ops.append(["store", best_evict])
                        on_disk.add(best_evict)
                    else:
                        ops.append(["delete", best_evict])
                    in_memory.remove(best_evict)
                    op_count += 1
                    continue
                    
            ops.append(["load", v_to_load])
            in_memory.add(v_to_load)
            if v_to_load in on_disk:
                on_disk.remove(v_to_load)
            op_count += 1
            
    return ops[:20000]

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 50
    try:
        result = _pebble_optimized(dag, fast_memory_budget, deadline)
        if result is not None and len(result) <= 20000:
            return result
    except Exception:
        pass
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    except Exception:
        parents = dag.get("parents", {})
        outputs = list(dag.get("outputs", []))
        vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
        ops = []
        for v in vertices:
            for p in parents.get(v, []):
                ops.append(["load", p])
            ops.append(["compute", v])
            if v in outputs:
                ops.append(["store", v])
        return ops[:20000]
# EVOLVE-BLOCK-END
