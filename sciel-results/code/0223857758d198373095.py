import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    dag structure expected:
      - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
      - dag['outputs']: list of output vertices
      - dag['inputs']: list of input vertices (start blue)
    Returns list of [operation, vertex] where operation in {load, store, compute, delete}.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Wall-clock limit with safety margin
    
    try:
        parents = dag.get('parents', {})
        outputs = set(dag.get('outputs', []))
        inputs = set(dag.get('inputs', []))
        
        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | inputs | outputs
        
        # Topological sort
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0
            
        queue = [v for v in all_vertices if in_degree.get(v, 0) == 0]
        topo_order = []
        while queue:
            # Use heap for deterministic order if needed, but list is fine
            queue.sort()
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
                    
        if len(topo_order) != len(all_vertices):
            # Cycle detected or invalid DAG, fallback
            return []
            
        topo_index = {v: i for i, v in enumerate(topo_order)}
        
        # Precompute last_use
        # last_use[v] is the max topo_index of any child of v.
        # If v is an output, last_use[v] is N (end of DAG).
        N = len(topo_order)
        last_use = {}
        for v in all_vertices:
            max_idx = -1
            for c in children.get(v, []):
                idx = topo_index.get(c, -1)
                if idx > max_idx:
                    max_idx = idx
            if v in outputs:
                last_use[v] = N
            else:
                last_use[v] = max_idx
                
        # Precompute children indices for faster next_use lookup
        # children_indices[v] = sorted list of topo_index of children of v
        children_indices = {}
        for v in all_vertices:
            idxs = [topo_index[c] for c in children.get(v, [])]
            idxs.sort()
            children_indices[v] = idxs

        def get_next_use(vertex, current_idx):
            """Find the topo index of the next child of vertex after current_idx."""
            idxs = children_indices.get(vertex, [])
            # Binary search or linear scan
            # Since lists are small, linear scan is fine
            for idx in idxs:
                if idx > current_idx:
                    return idx
            return float('inf')

        ops = []
        in_memory = set()
        
        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE:
                break
                
            # 1. Ensure v is in memory
            if v in inputs:
                if v not in in_memory:
                    # Need to load v
                    if len(in_memory) >= fast_memory_budget:
                        # Evict
                        # We cannot evict v itself (not in memory yet)
                        # We cannot evict any vertex that is a parent of v? 
                        # No, v is an input, so it has no parents.
                        # So we can evict anyone in memory.
                        candidates = list(in_memory)
                        if candidates:
                            # Evict the one with furthest next use
                            best_q = None
                            best_next = -1
                            for q in candidates:
                                nu = get_next_use(q, i)
                                if nu > best_next:
                                    best_next = nu
                                    best_q = q
                            if best_q is not None:
                                ops.append(['delete', best_q])
                                in_memory.discard(best_q)
                    ops.append(['load', v])
                    in_memory.add(v)
            else:
                # v is not an input. It must be computed.
                # Ensure all parents are in memory
                ps = parents.get(v, [])
                for p in ps:
                    if p not in in_memory:
                        if len(in_memory) >= fast_memory_budget:
                            # Evict
                            # Cannot evict any parent of v (needed for compute)
                            parent_set = set(ps)
                            candidates = [q for q in in_memory if q not in parent_set]
                            if candidates:
                                best_q = None
                                best_next = -1
                                for q in candidates:
                                    nu = get_next_use(q, i)
                                    if nu > best_next:
                                        best_next = nu
                                        best_q = q
                                if best_q is not None:
                                    ops.append(['delete', best_q])
                                    in_memory.discard(best_q)
                            else:
                                # Cannot evict, stuck.
                                # This should not happen if budget >= max fan-in
                                pass
                        
                        ops.append(['load', p])
                        in_memory.add(p)
                
                # Compute v
                ops.append(['compute', v])
                in_memory.add(v)
                
            # 2. Delete vertices that are no longer needed
            # A vertex u can be deleted if:
            # - u is not an output
            # - last_use[u] == i (i.e., v is the last child of u)
            #   Note: last_use[u] is the index of the last child.
            #   If last_use[u] == i, then the last child is v.
            #   So after processing v, u is no longer needed.
            
            # Iterate over a copy of in_memory
            for u in list(in_memory):
                if u in outputs:
                    continue
                if last_use[u] == i:
                    ops.append(['delete', u])
                    in_memory.discard(u)
                    
        # Ensure outputs are in memory
        # They should be, but just in case
        for o in outputs:
            if o not in in_memory:
                # This implies we deleted it or never loaded it?
                # If we never loaded it, it's an error in DAG.
                # If we deleted it, we made a mistake.
                # Try to load it.
                if len(in_memory) < fast_memory_budget:
                    ops.append(['load', o])
                    in_memory.add(o)
                else:
                    # Evict someone non-output
                    candidates = [q for q in in_memory if q not in outputs]
                    if candidates:
                        best_q = candidates[0] # Arbitrary
                        ops.append(['delete', best_q])
                        in_memory.discard(best_q)
                        ops.append(['load', o])
                        in_memory.add(o)
                    
        return ops

    except Exception:
        return []
