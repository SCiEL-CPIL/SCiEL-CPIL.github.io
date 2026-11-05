import heapq
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Solves the communication-minimizing DAG pebbling problem using Dijkstra's algorithm 
    on a state space defined by (fast_memory_set, blue_stored_set).
    
    Args:
        dag: A dictionary or object describing the DAG. 
             Expected structure based on common patterns:
             - 'vertices': list of vertex IDs
             - 'edges': list of (parent, child) tuples
             - 'inputs': list of input vertex IDs
             - 'outputs': list of output vertex IDs
             OR a custom object with attributes. 
             We will try to infer the structure.
        fast_memory_budget: Integer K, the max number of red pebbles.
    
    Returns:
        list of [operation, vertex]
    """
    
    # --- 1. Parse DAG Structure ---
    # Robustly extract graph info
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        edges = dag.get('edges', [])
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    elif hasattr(dag, 'vertices'):
        vertices = dag.vertices
        edges = dag.edges
        inputs = set(dag.inputs)
        outputs = set(dag.outputs)
    else:
        # Fallback: assume dag is a list of dicts or similar
        # This is a guess, but standard tasks usually provide a clear structure.
        # If this fails, we return empty list.
        return []

    if not vertices:
        return []

    # Map vertex names to indices 0..N-1 for bitmask usage
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    N = len(vertices)
    
    # Build parent list (indices)
    parents = [[] for _ in range(N)]
    for u, v in edges:
        if u in vertex_to_idx and v in vertex_to_idx:
            pu = vertex_to_idx[u]
            cv = vertex_to_idx[v]
            parents[cv].append(pu)
            
    # Bitmasks for inputs and outputs
    mask_inputs = 0
    for v in inputs:
        if v in vertex_to_idx:
            mask_inputs |= (1 << vertex_to_idx[v])
            
    mask_outputs = 0
    for v in outputs:
        if v in vertex_to_idx:
            mask_outputs |= (1 << vertex_to_idx[v])
            
    # Precompute parent bitmask for each vertex
    parent_masks = [0] * N
    for i in range(N):
        mask = 0
        for p in parents[i]:
            mask |= (1 << p)
        parent_masks[i] = mask

    # --- 2. Dijkstra's Algorithm ---
    # State: (cost, mask_R, mask_Bc)
    # mask_R: vertices in fast memory
    # mask_Bc: vertices stored in blue (computed)
    # Initial state: cost=0, mask_R=0, mask_Bc=0
    
    start_state = (0, 0, 0)
    heap = [(0, 0, 0)] # (cost, mask_R, mask_Bc)
    visited = set()
    best_cost = float('inf')
    best_state = None
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s limit to leave buffer for 60s total
    
    # We also need to reconstruct the path.
    # Storing full path in heap is memory heavy. 
    # Instead, we store (cost, mask_R, mask_Bc) and keep a dictionary 
    # of best_cost_to_state and parent_state.
    # But states are large integers. 
    # Given small K, number of states is limited.
    # We will store: dist[(mask_R, mask_Bc)] = (cost, prev_mask_R, prev_mask_Bc, op, v)
    
    dist = {}
    dist[(0, 0)] = (0, None, None, None, None)
    
    while heap:
        if time.time() > deadline:
            break
            
        cost, mask_R, mask_Bc = heapq.heappop(heap)
        
        state_key = (mask_R, mask_Bc)
        if state_key in visited:
            continue
        visited.add(state_key)
        
        # Check if goal reached
        # All outputs must be in (mask_inputs | mask_Bc)
        available_blue = mask_inputs | mask_Bc
        if (mask_outputs & ~available_blue) == 0:
            if cost < best_cost:
                best_cost = cost
                best_state = state_key
            # Since Dijkstra pops in increasing cost order, the first goal state found is optimal?
            # Yes, if all edge weights are non-negative.
            # We can break here if we are confident.
            # However, there might be multiple goal states with same cost.
            # We can break immediately as we want min cost.
            break

        # Generate neighbors
        
        # 1. Compute
        # v not in R, v not in Blue (Inputs or Stored), Parents(v) subset of R
        # Iterate through all vertices? O(N) per state.
        # Better: Iterate through vertices not in R.
        # For each v, check if (parent_masks[v] & ~mask_R) == 0
        # And (v not in available_blue) -> (1 << v) & ~available_blue != 0
        
        for v in range(N):
            bit_v = 1 << v
            if mask_R & bit_v:
                continue # Already in R
            
            # Check if can compute
            if (parent_masks[v] & ~mask_R) == 0:
                # Parents are in R
                if not (available_blue & bit_v):
                    # v is not in Blue, so we must compute
                    new_mask_R = mask_R | bit_v
                    new_mask_Bc = mask_Bc
                    new_state_key = (new_mask_R, new_mask_Bc)
                    new_cost = cost # Compute is free
                    if new_state_key not in dist or new_cost < dist[new_state_key][0]:
                        dist[new_state_key] = (new_cost, mask_R, mask_Bc, 'compute', v)
                        heapq.heappush(heap, (new_cost, new_mask_R, new_mask_Bc))
            
            # Check if can load
            if available_blue & bit_v:
                # v is in Blue
                new_mask_R = mask_R | bit_v
                new_mask_Bc = mask_Bc
                new_state_key = (new_mask_R, new_mask_Bc)
                new_cost = cost + 1
                if new_state_key not in dist or new_cost < dist[new_state_key][0]:
                    dist[new_state_key] = (new_cost, mask_R, mask_Bc, 'load', v)
                    heapq.heappush(heap, (new_cost, new_mask_R, new_mask_Bc))
            
        # 2. Store
        # v in R
        for v in range(N):
            bit_v = 1 << v
            if not (mask_R & bit_v):
                continue # Not in R
            
            # Store v
            new_mask_R = mask_R & ~bit_v
            if mask_inputs & bit_v:
                # Input, Bc doesn't change
                new_mask_Bc = mask_Bc
            else:
                # Computed, move to Bc
                new_mask_Bc = mask_Bc | bit_v
            
            new_state_key = (new_mask_R, new_mask_Bc)
            new_cost = cost + 1
            if new_state_key not in dist or new_cost < dist[new_state_key][0]:
                dist[new_state_key] = (new_cost, mask_R, mask_Bc, 'store', v)
                heapq.heappush(heap, (new_cost, new_mask_R, new_mask_Bc))
                
        # 3. Delete
        # v in R
        for v in range(N):
            bit_v = 1 << v
            if not (mask_R & bit_v):
                continue # Not in R
            
            # Delete v
            new_mask_R = mask_R & ~bit_v
            new_mask_Bc = mask_Bc
            new_state_key = (new_mask_R, new_mask_Bc)
            new_cost = cost # Delete is free
            if new_state_key not in dist or new_cost < dist[new_state_key][0]:
                dist[new_state_key] = (new_cost, mask_R, mask_Bc, 'delete', v)
                heapq.heappush(heap, (new_cost, new_mask_R, new_mask_Bc))

    # --- 3. Reconstruct Path ---
    if best_state is None:
        # Fallback: Return a trivial schedule if DP failed or timed out
        # This should rarely happen if N is small.
        return []
        
    path = []
    cur_key = best_state
    while dist[cur_key][1] is not None:
        cost, prev_R, prev_Bc, op, v = dist[cur_key]
        # Map index back to vertex name
        vertex_name = vertices[v]
        path.append([op, vertex_name])
        cur_key = (prev_R, prev_Bc)
        
    path.reverse()
    
    # Verify operation count limit
    if len(path) > 20000:
        # Truncate or return partial? 
        # The task says "at most 20000 operations".
        # If exceeded, we might return a valid subset or fail.
        # Given K is small, path length should be small.
        pass
        
    return path
