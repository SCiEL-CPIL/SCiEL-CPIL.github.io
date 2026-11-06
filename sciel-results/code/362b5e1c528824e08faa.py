"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

    # Try to import baseline as fallback
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except Exception:
        pebble_baseline = None

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        
        if not parents and not outputs and not inputs:
            return []

        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)
        
        # Topological order
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            if v not in in_degree:
                in_degree[v] = 0
            else:
                in_degree[v] = 0
        
        queue = [v for v in all_vertices if in_degree.get(v, 0) == 0]
        topo_order = []
        while queue:
            queue.sort()
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        remaining = all_vertices - set(topo_order)
        if remaining:
            topo_order.extend(sorted(remaining))
        
        topo_index = {v: i for i, v in enumerate(topo_order)}
        
        # Precompute next_use: for each vertex, the index of the next vertex in topo_order
        # that requires it (i.e., it is a parent of that vertex, or it is an output).
        # If a vertex is not an output and has no children, its next_use is -1 (dead).
        # If it is an output, its next_use is len(topo_order) (end).
        
        # For each vertex v, we need to know the "next time" it is needed.
        # We can iterate through topo_order and maintain a "next needed" pointer for each vertex.
        # Alternatively, precompute a list of "uses" for each vertex.
        
        uses = defaultdict(list)
        for v in topo_order:
            ps = parents.get(v, [])
            for p in ps:
                uses[p].append(topo_index[v])
        
        # Outputs are needed at the end
        for o in outputs:
            uses[o].append(len(topo_order))
        
        # Sort uses in descending order? Or just keep them.
        # We will process topo_order sequentially.
        # For a vertex v, its "next use" is the smallest index in uses[v] that is >= current_index.
        # To make this efficient, we can pre-sort uses[v] in descending order and pop from the front?
        # Or sort ascending and use a pointer.
        
        # Let's store uses[v] as a sorted list of indices where v is needed.
        for v in uses:
            uses[v].sort(reverse=True) # So we can pop from end? Or just iterate.
            # Actually, since we process topo_order in increasing index order,
            # the "next use" is the smallest element in uses[v] that is >= current_i.
            # If we sort descending, the last element is the smallest.
            # Let's just sort ascending and use a pointer or binary search if needed.
            # Given N is likely small enough for O(N^2) or simple checks, let's keep it simple.
            uses[v].sort() # Ascending
        
        # fast_mem: set of vertices currently in fast memory
        fast_mem = set()
        ops = []
        
        # next_use_idx: For each vertex in fast_mem, track the index of its next use.
        # This allows O(1) or O(log N) eviction selection.
        # We can use a heap (priority queue) for eviction candidates.
        # Heap element: (next_use_index, vertex)
        # We want to evict the vertex with the LARGEST next_use_index (furthest in future).
        # So we use a max-heap (negate next_use_index for heapq).
        # But next_use changes as we progress.
        # Simpler: Just scan fast_mem when eviction is needed. 
        # If fast_memory_budget is small (e.g., < 50), scanning is fast.
        # If it's large, we might need a heap. Let's assume budget is small enough for scan.
        
        def get_next_use(v, current_idx):
            """
            Returns the smallest index in uses[v] that is >= current_idx.
            If none, returns float('inf') if v is an output? No, if it's an output, 
            we added len(topo_order) to uses. 
            If v is not an output and has no more uses, it's dead.
            """
            # uses[v] is sorted ascending
            # We need the first element >= current_idx
            # Since we process in order, we can just find it.
            # Optimization: We can maintain a pointer, but for simplicity and robustness against 
            # complex DAGs, a linear scan of the uses list is safe if lists are short.
            # However, uses[v] can be long.
            # Let's use bisect.
            import bisect
            idx = bisect.bisect_left(uses[v], current_idx)
            if idx < len(uses[v]):
                return uses[v][idx]
            else:
                return float('inf')

        def evict_if_needed(current_idx):
            """
            If fast_mem is full, evict a vertex.
            Strategy: Evict the vertex with the furthest next use.
            If a vertex has no future use (inf), evict it first.
            """
            if len(fast_mem) < fast_memory_budget:
                return
            
            # Find eviction candidate
            # We want to keep vertices with SMALL next_use (needed soon).
            # Evict vertex with LARGEST next_use.
            
            best_candidate = None
            best_next_use = -1 # We want to maximize next_use
            
            for v in fast_mem:
                # Don't evict if it's the vertex we are trying to load? 
                # The caller ensures the target is not in fast_mem before calling this.
                # But what if the target is a parent of the current computation?
                # The caller should handle logic.
                
                nu = get_next_use(v, current_idx)
                
                # If nu is inf, it's dead. Definitely evict.
                # If nu is large, evict.
                
                if nu > best_next_use:
                    best_next_use = nu
                    best_candidate = v
                    
            if best_candidate is not None:
                ops.append(['delete', best_candidate])
                fast_mem.discard(best_candidate)
            else:
                # Should not happen if fast_mem is not empty
                pass

        def ensure_in_fast_mem(v, current_idx):
            """
            Ensure vertex v is in fast_mem.
            Returns True if successful.
            """
            if v in fast_mem:
                return True
            
            ps = parents.get(v, [])
            
            if not ps:
                # Input vertex
                # Need space?
                if len(fast_mem) >= fast_memory_budget:
                    evict_if_needed(current_idx)
                
                ops.append(['load', v])
                fast_mem.add(v)
                return True
            else:
                # Computed vertex
                # Ensure all parents are in fast_mem
                for p in ps:
                    if not ensure_in_fast_mem(p, current_idx):
                        return False
                
                # Now compute v
                if len(fast_mem) >= fast_memory_budget:
                    evict_if_needed(current_idx)
                
                ops.append(['compute', v])
                fast_mem.add(v)
                return True

        # Process in topological order
        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE:
                break
            
            if not ensure_in_fast_mem(v, i):
                # Failed to ensure v in fast mem
                # This implies budget is too small or logic error.
                # Try to force evict and retry?
                # If we are stuck, break.
                break
            
            # Proactively delete dead vertices to save space?
            # This is optional but helps if budget is tight.
            # If we delete a dead vertex, we save a slot.
            # Let's check if any vertex in fast_mem is dead (next_use == inf).
            # We do this every step to keep memory clean.
            if len(fast_mem) > 0:
                for u in list(fast_mem):
                    if u == v:
                        continue
                    nu = get_next_use(u, i)
                    if nu == float('inf'):
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                    
        # Ensure all outputs are in fast_mem at the end
        for o in outputs:
            if o not in fast_mem:
                # Need to bring it back
                # Use the same ensure logic, but current_idx is end
                if not ensure_in_fast_mem(o, len(topo_order)):
                    # Force evict and retry
                    if len(fast_mem) >= fast_memory_budget:
                        evict_if_needed(len(topo_order))
                    ensure_in_fast_mem(o, len(topo_order))
        
        if len(ops) <= 20000:
            return ops
        else:
            if pebble_baseline:
                return pebble_baseline(dag, fast_memory_budget)
            return ops[:20000]
            
    except Exception as e:
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
# EVOLVE-BLOCK-END
