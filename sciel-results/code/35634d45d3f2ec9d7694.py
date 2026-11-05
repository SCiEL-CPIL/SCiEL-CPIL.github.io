"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store ops) for DAG pebbling.
    
    dag structure assumed:
    - dag['vertices']: list of vertex IDs
    - dag['edges']: list of (parent, child) or similar
    - dag['inputs']: list of input vertex IDs
    - dag['outputs']: list of output vertex IDs
    - dag['topo_order']: topological ordering
    
    We'll try to infer the structure and use A* for small instances,
    greedy for larger ones.
    """
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Try to extract DAG structure
    try:
        vertices = dag.get('vertices', dag.get('nodes', []))
        if not vertices:
            if isinstance(dag, dict):
                for key in ['V', 'verts', 'ids', 'nodes']:
                    if key in dag:
                        vertices = dag[key]
                        break
            if not vertices:
                edges = dag.get('edges', dag.get('E', []))
                vset = set()
                for e in edges:
                    if isinstance(e, (list, tuple)) and len(e) == 2:
                        vset.add(e[0])
                        vset.add(e[1])
                vertices = sorted(vset)
            else:
                vertices = list(vertices)
        
        edges = dag.get('edges', dag.get('E', []))
        inputs = dag.get('inputs', dag.get('input', []))
        outputs = dag.get('outputs', dag.get('output', []))
        topo_order = dag.get('topo_order', dag.get('topo', dag.get('order', None)))
        
        if not inputs:
            has_incoming = set()
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    has_incoming.add(e[1])
            inputs = [v for v in vertices if v not in has_incoming]
        
        if not outputs:
            has_outgoing = set()
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    has_outgoing.add(e[0])
            outputs = [v for v in vertices if v not in has_outgoing]
        
        if not topo_order:
            indegree = {v: 0 for v in vertices}
            children = {v: [] for v in vertices}
            parents = {v: [] for v in vertices}
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    p, c = e[0], e[1]
                    if p in indegree and c in indegree:
                        indegree[c] += 1
                        children[p].append(c)
                        parents[c].append(p)
            
            queue = deque([v for v in vertices if indegree[v] == 0])
            topo_order = []
            while queue:
                v = queue.popleft()
                topo_order.append(v)
                for c in children[v]:
                    indegree[c] -= 1
                    if indegree[c] == 0:
                        queue.append(c)
            
            if len(topo_order) != len(vertices):
                remaining = set(vertices) - set(topo_order)
                topo_order.extend(sorted(remaining))
        
        vertices = list(vertices)
        inputs = list(inputs)
        outputs = list(outputs)
        topo_order = list(topo_order)
        
        # Build parent/child maps
        children_map = {v: [] for v in vertices}
        parents_map = {v: [] for v in vertices}
        for e in edges:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                p, c = e[0], e[1]
                if p in children_map and c in parents_map:
                    children_map[p].append(c)
                    parents_map[c].append(p)
        
        # Index vertices for bitmask
        v_to_idx = {v: i for i, v in enumerate(vertices)}
        n = len(vertices)
        
        input_mask = 0
        for v in inputs:
            if v in v_to_idx:
                input_mask |= (1 << v_to_idx[v])
        
        output_mask = 0
        for v in outputs:
            if v in v_to_idx:
                output_mask |= (1 << v_to_idx[v])
        
        # For A*: state = (loaded_mask, computed_mask)
        # loaded_mask: vertices currently in fast memory (blue)
        # computed_mask: vertices that have been computed at least once (can be recomputed)
        # Actually, let's think differently:
        # State: which vertices are in fast memory
        # A vertex can be computed if all its parents are in fast memory
        # After computing, the result is in fast memory
        # We can store (remove from fast memory) any non-input vertex
        # We can load any vertex that has been computed before (recomputation allowed)
        
        # Wait - "Recomputations allowed" means we can recompute a vertex from its parents
        # even if we stored it. So the state just needs to track what's in fast memory.
        
        # Naive schedule cost: for each non-input vertex, we need to load each parent once
        # (or keep them), compute, then maybe store. Let's estimate naive cost.
        
        # Let's use a simpler approach: greedy with memory management
        # Process vertices in topological order
        # Keep track of what's in fast memory
        # When computing a vertex, ensure all parents are in fast memory
        # If not, load them (cost 1 each)
        # After computing, if memory is full, store some vertices
        
        # Actually, let me try A* for small n
        if n <= 15 and time.time() < deadline - 10:
            result = _astar_pebble(n, input_mask, output_mask, parents_map, v_to_idx, vertices, fast_memory_budget, deadline)
            if result is not None:
                return result
        
        # Greedy fallback
        result = _greedy_pebble(n, input_mask, output_mask, parents_map, children_map, v_to_idx, vertices, topo_order, fast_memory_budget, deadline)
        if result is not None:
            return result
        
    except Exception as e:
        pass
    
    # Ultimate fallback: try baseline
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    except:
        return []

def _astar_pebble(n, input_mask, output_mask, parents_map, v_to_idx, vertices, budget, deadline):
    """A* search over memory configurations. State: bitmask of vertices in fast memory."""
    start_time = time.time()
    
    # State: (loaded_mask) - which vertices are in fast memory
    # Initial: inputs are loaded
    # Goal: all outputs are loaded
    
    # For a vertex to be computable, all its parents must be in fast memory
    # After computing vertex v, v is added to fast memory
    # We can remove (store) any non-input vertex from fast memory
    # Cost: load=1, store=1, compute=0 (no transfer), delete=0? 
    
    # Wait, the operations are load/store/compute/delete
    # load: bring a value from slow to fast (cost 1 transfer)
    # store: move from fast to slow (cost 1 transfer)  
    # compute: compute a vertex from its parents (cost 0, but requires parents in fast)
    # delete: remove from fast without storing? (cost 0)
    
    # So transfers = number of load + store operations
    
    # A* heuristic: minimum additional loads needed to get all outputs
    # Admissible: for each output not yet in memory, we need at least 1 load
    # Better: for each output, if its parents aren't all in memory, we need loads
    
    # Let's use a simpler heuristic: number of outputs not in fast memory
    # This is admissible since each such output needs at least one load (or compute chain)
    
    # Actually, compute is free, so if we can compute the output from parents in memory, no load needed
    # The heuristic should be: minimum number of loads needed assuming optimal compute order
    
    # For simplicity, use h = number of output vertices not in current fast memory
    # This is admissible because each such vertex must be either loaded or computed
    # and if computed, its parents must be in memory, which may require loads
    
    # Let's just use h = 0 for now (Dijkstra) to guarantee optimality, with pruning
    
    # State space: 2^n states, too many for n>15
    # Use A* with h = popcount(output_mask & ~loaded_mask)
    
    def heuristic(loaded_mask):
        # Count outputs not in loaded_mask
        missing_outputs = output_mask & ~loaded_mask
        return bin(missing_outputs).count('1')
    
    # Priority queue: (f, g, loaded_mask, operations)
    # f = g + h
    # g = number of transfers so far
    
    start_state = input_mask
    if start_state & output_mask == output_mask:
        # All outputs already in memory
        return []
    
    open_set = []
    h0 = heuristic(start_state)
    heapq.heappush(open_set, (h0, 0, start_state, []))
    
    best_g = {start_state: 0}
    
    max_expansions = 50000
    expansions = 0
    
    while open_set and expansions < max_expansions:
        if time.time() > deadline - 5:
            return None
        
        f, g, state, ops = heapq.heappop(open_set)
        
        if g > best_g.get(state, float('inf')):
            continue
        
        # Check goal
        if (state & output_mask) == output_mask:
            return ops
        
        expansions += 1
        
        # Generate successors
        # 1. Compute a vertex: if all parents are in state, and vertex not in state
        #    New state: state | (1 << v), cost 0
        # 2. Store a vertex: if vertex is in state and not an input
        #    New state: state & ~(1 << v), cost 1
        # 3. Load a vertex: if vertex is not in state
        #    New state: state | (1 << v), cost 1
        #    (Loading any vertex is allowed - recomputation means we can load computed values)
        
        # For compute: find vertices whose parents are all in state
        for v in range(n):
            v_bit = 1 << v
            if state & v_bit:
                continue  # Already in memory
            
            # Check if all parents are in state
            v_name = vertices[v]
            parents = parents_map.get(v_name, [])
            all_parents_loaded = True
            for p in parents:
                if p in v_to_idx:
                    p_bit = 1 << v_to_idx[p]
                    if not (state & p_bit):
                        all_parents_loaded = False
                        break
            
            if all_parents_loaded:
                new_state = state | v_bit
                new_g = g  # compute is free
                if new_g < best_g.get(new_state, float('inf')):
                    best_g[new_state] = new_g
                    new_ops = ops + [['compute', v_name]]
                    hf = heuristic(new_state)
                    heapq.heappush(open_set, (new_g + hf, new_g, new_state, new_ops))
        
        # For store: remove a non-input vertex from memory
        input_set = set()
        for v in range(n):
            v_name = vertices[v]
            # Check if v is an input
            # We need to know which vertices are inputs
            # Let's pass this info... actually we have input_mask
            v_bit = 1 << v
            if state & v_bit:
                if not (input_mask & v_bit):  # Not an input
                    new_state = state & ~v_bit
                    new_g = g + 1
                    if new_g < best_g.get(new_state, float('inf')):
                        best_g[new_state] = new_g
                        new_ops = ops + [['store', v_name]]
                        hf = heuristic(new_state)
                        heapq.heappush(open_set, (new_g + hf, new_g, new_state, new_ops))
        
        # For load: add any vertex not in memory
        # This is expensive in terms of branching, so limit it
        # Only load vertices that are ancestors of outputs (useful vertices)
        # For simplicity, skip load in A* and rely on compute + store
        # Actually, load is needed for recomputation after store
        # Let's include load but with a limit
        
        # Skip load for now to keep branching factor manageable
        # Load will be handled by the greedy fallback if needed
    
    return None

def _greedy_pebble(n, input_mask, output_mask, parents_map, children_map, v_to_idx, vertices, topo_order, budget, deadline):
    """Greedy pebbling: process in topological order, manage memory."""
    start_time = time.time()
    
    # Track what's in fast memory
    in_memory = set()
    for v in range(n):
        if input_mask & (1 << v):
            in_memory.add(vertices[v])
    
    # Track what has been computed (for potential reload)
    computed = set()
    
    operations = []
    transfer_count = 0
    
    # Process vertices in topological order
    # For each vertex, ensure all parents are in memory, then compute
    for v in topo_order:
        if time.time() > deadline - 5:
            break
        
        if v in in_memory:
            computed.add(v)
            continue
        
        # Check if all parents are in memory
        parents = parents_map.get(v, [])
        missing_parents = [p for p in parents if p not in in_memory]
        
        if missing_parents:
            # Need to load missing parents
            # Prioritize loading parents that will be needed again
            # For now, just load all missing parents
            for p in missing_parents:
                if p not in in_memory:
                    # Load p
                    operations.append(['load', p])
                    in_memory.add(p)
                    transfer_count += 1
            
            # Now compute v
            operations.append(['compute', v])
            in_memory.add(v)
            computed.add(v)
        else:
            # All parents in memory, compute v
            operations.append(['compute', v])
            in_memory.add(v)
            computed.add(v)
        
        # Memory management: if over budget, store some vertices
        if len(in_memory) > budget:
            # Store vertices that won't be needed again
            # Count future uses
            future_uses = {}
            for u in topo_order[topo_order.index(v)+1:]:
                for p in parents_map.get(u, []):
                    future_uses[p] = future_uses.get(p, 0) + 1
            
            # Also count direct output needs
            for out in output_mask:
                pass  # We don't have output list here, but outputs are in topo_order
            
            # Store vertices with 0 future uses, preferring those not in computed set that are inputs
            to_store = []
            for mem_v in list(in_memory):
                if mem_v not in future_uses and mem_v not in set(vertices[v] for v in range(n) if output_mask & (1 << v_to_idx.get(mem_v, -1))):
                    to_store.append(mem_v)
            
            # Sort by least likely to be needed
            to_store.sort(key=lambda x: future_uses.get(x, 0))
            
            while len(in_memory) > budget and to_store:
                to_remove = to_store.pop(0)
                if to_remove in in_memory:
                    operations.append(['store', to_remove])
                    in_memory.remove(to_remove)
                    transfer_count += 1
            
            # If still over, store anything (including inputs, but they can be reloaded)
            if len(in_memory) > budget:
                for mem_v in list(in_memory):
                    if len(in_memory) <= budget:
                        break
                    operations.append(['store', mem_v])
                    in_memory.remove(mem_v)
                    transfer_count += 1
    
    # Ensure all outputs are in memory at the end
    for v in range(n):
        if output_mask & (1 << v):
            v_name = vertices[v]
            if v_name not in in_memory:
                operations.append(['load', v_name])
                in_memory.add(v_name)
                transfer_count += 1
    
    return operations
# EVOLVE-BLOCK-END
