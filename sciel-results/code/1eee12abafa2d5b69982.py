"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy:
    - Topological order the DAG.
    - Use a "window" of fast memory to hold intermediate results.
    - When fast memory is full, we must either:
        a) Delete some vertex (if it's not needed later, or can be recomputed)
        b) Load from slow memory (if it was stored)
    - We aim to minimize transfers (load/store operations).
    
    Key insight: We can use a simple heuristic:
    - Process vertices in topological order.
    - Keep a set of vertices currently in fast memory.
    - When computing a vertex, if its parents are not in fast memory, load them.
    - After computing, if fast memory is full, we need to evict a vertex.
      - Prefer to evict vertices that are not needed by any unprocessed vertex.
      - If all are needed, evict the one with the highest recompute cost (or just pick one).
    
    Actually, a better approach for minimizing transfers:
    - Use a recursive/pebbling strategy. But that's complex.
    
    Let's try a simpler, effective heuristic:
    1. Topologically sort the DAG.
    2. For each vertex in topological order:
       a. Ensure all parents are in fast memory (load if not, from slow memory or recompute).
       b. Compute the vertex.
       c. If the vertex is an output, it must remain in fast memory until the end? 
          No, outputs must finish blue, but can be stored to slow memory.
       d. If fast memory is full, evict a vertex. 
          - We can evict a vertex only if it's not an input (inputs start blue, but can be reloaded).
          - Actually, inputs are always available in slow memory.
          - We should evict the vertex that is needed least often in the future.
    
    Let's define:
    - dag: a dict or object with:
        - 'parents': dict mapping vertex -> list of parent vertices
        - 'outputs': list of output vertices
        - 'inputs': list of input vertices (or we can infer)
    
    Since the exact structure of 'dag' is not specified, I'll assume a common format:
    - dag is a dict with keys:
        - 'vertices': list of all vertices
        - 'parents': dict {vertex: [parent1, parent2, ...]}
        - 'outputs': list of output vertices
        - 'inputs': list of input vertices (optional, can be inferred as vertices with no parents)
    
    However, the task says "dag has topologically ordered parents", so parents are given in topo order.
    
    Let's assume:
    - dag['vertices'] is a list of vertices in topological order.
    - dag['parents'] is a dict mapping each vertex to its list of parents.
    - dag['outputs'] is a list of output vertices.
    
    Algorithm:
    1. Parse the DAG.
    2. Compute in-degree and reverse edges for future need analysis.
    3. Process vertices in topological order.
    4. Maintain:
        - fast_mem: set of vertices currently in fast memory
        - slow_mem: dict of vertices that have been computed and stored in slow memory (value not needed, just presence)
        - operations: list of (operation, vertex)
    
    For each vertex v in topo order:
      a. If v is an input:
         - It's already blue in slow memory.
         - If fast_mem is full, we can't load it without evicting. But inputs are always available.
         - Actually, we don't need to "load" inputs unless we want them in fast memory for computation.
      b. Ensure all parents of v are in fast_mem:
         - For each parent p:
           - If p not in fast_mem:
             - If p in slow_mem: load p (operation: load, p)
             - Else: recompute p? This is complex. Better to store all computed vertices in slow_mem.
      c. Compute v (operation: compute, v). Now v is in fast_mem.
      d. If v is not an output and not needed by any future vertex, we can delete it to free space.
         - But we might need it if it's a parent of a future vertex.
      e. If len(fast_mem) > fast_memory_budget:
         - Evict a vertex. Choose the one with the fewest future uses.
         - Store it in slow_mem (operation: store, vertex) before deleting.
    
    Actually, let's refine:
    - We only store in slow memory when we evict.
    - We load from slow memory when we need a parent that's not in fast memory.
    
    Steps for each vertex v in topo order:
    1. For each parent p of v:
       - If p not in fast_mem:
         - If p in slow_mem:
           - operations.append(('load', p))
           - fast_mem.add(p)
         - Else:
           # This should not happen if we store all evicted vertices.
           # But if p was never computed? Inputs are special.
           if p in inputs:
             operations.append(('load', p))
             fast_mem.add(p)
           else:
             # Error: parent not available
             raise ValueError(f"Parent {p} not available")
    2. operations.append(('compute', v))
       fast_mem.add(v)
       slow_mem.add(v)  # Mark as available in slow memory (in case we evict and need to reload)
       # Actually, we only add to slow_mem when we store. But for inputs, they are always in slow_mem.
    3. If len(fast_mem) > fast_memory_budget:
       # Need to evict one vertex.
       # Choose the vertex w in fast_mem that has the minimum number of future children.
       # Future children: children of w that are not yet processed.
       # We can precompute for each vertex the list of children and their position in topo order.
       
       # Let's precompute:
       # children[v] = list of children of v
       # pos[v] = position of v in topo order
       
       # For each candidate w in fast_mem:
       #   future_children = [c for c in children[w] if pos[c] > pos[v]]
       #   If len(future_children) == 0, w is safe to evict.
       #   Otherwise, count the number of future uses.
       
       # Evict the one with the fewest future uses. If tie, evict the one with the largest index (arbitrary).
       
       # Before evicting, store w in slow memory.
       operations.append(('store', w))
       fast_mem.remove(w)
       # w is already in slow_mem (we added it when computed or it's an input)
    
    4. If v is an output, we don't evict it immediately? But we can, as long as we can reload it.
       # Actually, outputs must finish blue. So at the end, all outputs must be in fast_mem.
       # So we need to ensure that at the end, all outputs are in fast_mem.
       
    After processing all vertices:
    - Ensure all outputs are in fast_mem. If not, load them from slow_mem.
    
    But wait, the problem says "all outputs must finish blue", meaning in fast memory at the end.
    
    So after the main loop, for each output o:
      if o not in fast_mem:
        operations.append(('load', o))
        fast_mem.add(o)
    
    This should work.
    
    Let's code it.
    """
    # Parse the DAG
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    outputs = dag.get('outputs', [])
    
    # If vertices is empty, return empty list
    if not vertices:
        return []
    
    # Create position map
    pos = {v: i for i, v in enumerate(vertices)}
    
    # Create children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Identify inputs (vertices with no parents)
    inputs = set(v for v in vertices if not parents.get(v, []))
    
    # Initialize
    fast_mem = set()
    slow_mem = set(inputs)  # Inputs are always in slow memory
    operations = []
    
    # Precompute future children count for each vertex
    # future_children_count[v] = number of children of v that appear after v in topo order
    # But we need this dynamically as we process.
    # Instead, we can compute on the fly.
    
    # Process vertices in topological order
    for v in vertices:
        # Step 1: Ensure all parents are in fast_mem
        for p in parents.get(v, []):
            if p not in fast_mem:
                if p in slow_mem:
                    operations.append(('load', p))
                    fast_mem.add(p)
                else:
                    # This should not happen
                    raise ValueError(f"Parent {p} not in slow_mem")
        
        # Step 2: Compute v
        operations.append(('compute', v))
        fast_mem.add(v)
        slow_mem.add(v)
        
        # Step 3: If fast_mem is over budget, evict a vertex
        if len(fast_mem) > fast_memory_budget:
            # Find the vertex to evict
            # Score each vertex in fast_mem by number of future children
            best_evict = None
            best_score = float('inf')
            
            for w in fast_mem:
                # Count future children
                future_count = 0
                for c in children[w]:
                    if pos[c] > pos[v]:
                        future_count += 1
                
                # If future_count is 0, it's a good candidate
                # Use future_count as score (lower is better)
                if future_count < best_score:
                    best_score = future_count
                    best_evict = w
            
            if best_evict is not None:
                operations.append(('store', best_evict))
                fast_mem.remove(best_evict)
                # best_evict is already in slow_mem
    
    # Step 4: Ensure all outputs are in fast_mem
    for o in outputs:
        if o not in fast_mem:
            if o in slow_mem:
                operations.append(('load', o))
                fast_mem.add(o)
            else:
                raise ValueError(f"Output {o} not in slow_mem")
    
    return operations
# EVOLVE-BLOCK-END
