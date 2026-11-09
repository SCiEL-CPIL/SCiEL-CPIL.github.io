"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using chain decomposition and round-robin scheduling.
    
    The DAG is expected to have:
    - Topologically ordered vertices
    - Parents defined for each vertex
    - Output vertices marked
    
    Operations: load, store, compute, delete
    Blue values are in fast memory; red values are in slow memory.
    fast_memory_budget limits the number of red (fast-memory) slots.
    """
    
    # Try to extract DAG structure from various possible formats
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', range(len(dag.get('edges', [])))))
        parents_map = dag.get('parents', {})
        outputs = set(dag.get('outputs', []))
        num_vertices = len(vertices) if vertices else len(parents_map)
        if not vertices:
            vertices = list(range(num_vertices))
    elif isinstance(dag, (list, tuple)):
        # Could be a list of (vertex, parents) tuples or just a list of parent lists
        if len(dag) > 0 and isinstance(dag[0], (list, tuple)):
            parents_map = {i: dag[i] for i in range(len(dag))}
            vertices = list(range(len(dag)))
            outputs = set([len(dag) - 1]) if len(dag) > 0 else set()
            num_vertices = len(dag)
        else:
            parents_map = {i: [] for i in range(len(dag))}
            vertices = list(dag)
            outputs = set()
            num_vertices = len(dag)
    else:
        # Assume it's a class with attributes
        parents_map = getattr(dag, 'parents', {})
        vertices = getattr(dag, 'vertices', list(range(len(parents_map))))
        outputs = set(getattr(dag, 'outputs', []))
        num_vertices = len(vertices) if vertices else len(parents_map)
        if not vertices:
            vertices = list(range(num_vertices))
    
    # Build adjacency: children[v] = list of children
    children_map = {v: [] for v in range(num_vertices)}
    for v in range(num_vertices):
        par = parents_map.get(v, [])
        if par is None:
            par = []
        for p in par:
            children_map[p].append(v)
    
    # Identify input vertices (no parents) and output vertices
    inputs = set()
    for v in range(num_vertices):
        if not parents_map.get(v, []):
            inputs.add(v)
    
    if not outputs:
        # Outputs are vertices with no children
        outputs = set(v for v in range(num_vertices) if not children_map[v])
    
    # Convert to vertex IDs for consistency
    inputs = set(v if isinstance(v, int) else v for v in inputs)
    outputs = set(v if isinstance(v, int) else v for v in outputs)
    
    ops = []
    fast_memory = set()  # Currently in fast memory (blue)
    slow_memory = set()  # Currently in slow memory (red)
    
    # Compute topological order (should already be given, but verify)
    in_degree = {v: len(parents_map.get(v, [])) for v in range(num_vertices)}
    topo_order = []
    queue = deque([v for v in range(num_vertices) if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != num_vertices:
        # Fallback to range
        topo_order = list(range(num_vertices))
    
    # Chain decomposition using greedy approach
    # Decompose into vertex-disjoint paths (chains)
    chains = []
    assigned = [False] * num_vertices
    
    # Use maximum matching on bipartite graph for chain decomposition
    # For simplicity, use greedy longest path first
    # But we'll use a simpler greedy: repeatedly find the longest unassigned chain
    
    # First, compute for each vertex the length of the longest chain starting from it
    # using topological order (reverse)
    longest_from = [0] * num_vertices
    # Process in reverse topological order
    for v in reversed(topo_order):
        if children_map[v]:
            longest_from[v] = 1 + max(longest_from[c] for c in children_map[v])
        else:
            longest_from[v] = 1
    
    # Greedily assign chains
    available = [v for v in range(num_vertices) if not assigned[v]]
    # Sort by longest_from descending to prefer longer chains
    available.sort(key=lambda v: -longest_from[v])
    
    while available:
        # Pick the vertex with highest longest_from that hasn't been assigned
        start = None
        for v in available:
            if not assigned[v]:
                start = v
                break
        
        if start is None:
            break
        
        # Build chain from start
        chain = [start]
        assigned[start] = True
        curr = start
        while children_map[curr]:
            # Find unassigned child
            next_v = None
            for c in children_map[curr]:
                if not assigned[c]:
                    next_v = c
                    break
            if next_v is None:
                break
            chain.append(next_v)
            assigned[next_v] = True
            curr = next_v
        
        chains.append(chain)
        # Remove assigned from available
        available = [v for v in available if not assigned[v]]
        available.sort(key=lambda v: -longest_from[v])
    
    # Ensure all vertices are covered
    all_assigned = set()
    for chain in chains:
        for v in chain:
            all_assigned.add(v)
    
    # Add remaining vertices as single-element chains
    for v in range(num_vertices):
        if v not in all_assigned:
            chains.append([v])
            all_assigned.add(v)
    
    # Round-robin scheduling across chains
    # Each chain is processed as a stream: we need to compute each element in order
    # For each chain, we need to ensure parents are available
    
    # Pre-compute the computation order within each chain
    # A vertex v in a chain can only be computed when all its parents are in fast memory
    # We'll use a worklist approach
    
    # Track which vertices are currently in fast memory
    in_fast = set()
    in_slow = set()
    computed = set()  # Vertices that have been computed at least once
    stored = set()  # Vertices that have been stored to slow memory
    
    op_count = 0
    max_ops = 20000
    
    # We need to compute all output vertices
    # Strategy: process chains in round-robin fashion
    # For each chain, advance the pointer as far as possible
    
    chain_pos = [0] * len(chains)  # Current position in each chain
    pending_loads = set()  # Vertices waiting to be loaded
    
    def can_compute(v):
        """Check if vertex v can be computed: all parents in fast memory."""
        for p in parents_map.get(v, []):
            if p not in in_fast:
                return False
        return True
    
    def compute_vertex(v):
        """Compute vertex v, returning operations."""
        nonlocal op_count, in_fast, in_slow, computed, stored
        ops_out = []
        
        # All parents must be in fast memory (blue)
        # Compute the value
        # The result is now in fast memory (blue)
        # But we need a fast memory slot!
        
        if len(in_fast) < fast_memory_budget:
            # Can store in fast memory directly
            in_fast.add(v)
            computed.add(v)
            op_count += 1
            ops_out.append(['compute', v])
        else:
            # Need to evict something first
            # We can't compute without a free slot
            # This means we need to store a value to slow memory first
            # Find a vertex in fast memory that we can afford to evict
            # Heuristic: prefer to keep vertices that have many pending children
            evicted = None
            best_score = -1
            for u in in_fast:
                # Score: number of uncomputed children
                num_pending_children = 0
                for c in children_map.get(u, []):
                    if c not in computed:
                        num_pending_children += 1
                if num_pending_children < best_score or evicted is None:
                    if num_pending_children == 0 or evicted is None:
                        best_score = num_pending_children
                        evicted = u
            
            if evicted is not None:
                # Store evicted to slow memory
                in_fast.remove(evicted)
                in_slow.add(evicted)
                stored.add(evicted)
                op_count += 1
                ops_out.append(['store', evicted])
                
                # Now compute
                in_fast.add(v)
                computed.add(v)
                op_count += 1
                ops_out.append(['compute', v])
            else:
                # No evictable vertex? Shouldn't happen if budget >= 1
                # Force evict any vertex
                if in_fast:
                    evicted = next(iter(in_fast))
                    in_fast.remove(evicted)
                    in_slow.add(evicted)
                    stored.add(evicted)
                    op_count += 1
                    ops_out.append(['store', evicted])
                
                in_fast.add(v)
                computed.add(v)
                op_count += 1
                ops_out.append(['compute', v])
        
        return ops_out
    
    def load_vertex(v):
        """Load vertex v from slow memory to fast memory."""
        nonlocal op_count, in_fast, in_slow
        ops_out = []
        
        # v must be in slow memory
        # We need a free fast memory slot
        if len(in_fast) < fast_memory_budget:
            in_slow.remove(v)
            in_fast.add(v)
            op_count += 1
            ops_out.append(['load', v])
        else:
            # Need to evict
            evicted = None
            for u in in_fast:
                # Evict a vertex that is already stored or has few pending children
                evicted = u
                break
            
            if evicted is not None:
                in_fast.remove(evicted)
                if evicted not in stored:
                    in_slow.add(evicted)
                    stored.add(evicted)
                    op_count += 1
                    ops_out.append(['store', evicted])
            
            in_slow.remove(v)
            in_fast.add(v)
            op_count += 1
            ops_out.append(['load', v])
        
        return ops_out
    
    # Main scheduling loop
    # Initialize: all input vertices start blue (in fast memory?)
    # The problem says "Inputs start blue" - blue means in fast memory
    # So all input vertices start in fast memory
    
    # But wait - if there are many inputs, they might exceed the budget
    # Let's assume inputs that exceed budget need to be stored to slow memory
    
    # Actually, re-reading: "Inputs start blue" means they begin in fast memory
    # "every red value consumes one fast-memory slot" - I think red is slow memory
    
    # Let me reconsider: blue = fast memory, red = slow memory
    # "Inputs start blue" = inputs start in fast memory
    # "all outputs must finish blue" = outputs must end in fast memory
    
    # Initialize fast memory with input vertices
    for v in sorted(inputs):
        if v in range(num_vertices):
            if len(in_fast) < fast_memory_budget:
                in_fast.add(v)
                computed.add(v)  # Inputs are already computed
            else:
                # Need to store excess inputs to slow memory
                in_slow.add(v)
                stored.add(v)
                op_count += 1
                ops.append(['store', v])
    
    # Process chains in round-robin fashion
    progress = True
    rounds = 0
    max_rounds = max_ops
    
    while progress and op_count < max_ops:
        progress = False
        rounds += 1
        
        for ci, chain in enumerate(chains):
            if op_count >= max_ops:
                break
            
            # Try to advance this chain
            while chain_pos[ci] < len(chain):
                v = chain[ci]
                
                if v in computed and v in in_fast:
                    # Already done and in fast memory
                    chain_pos[ci] += 1
                    continue
                
                if v in computed and v not in in_fast:
                    # Computed but in slow memory, need to load
                    if v in in_slow:
                        if can_fit_in_fast():
                            ops.extend(load_vertex(v))
                            progress = True
                        else:
                            # Need to make room
                            # Try to store a low-priority vertex
                            if make_room_for_load(v):
                                ops.extend(load_vertex(v))
                                progress = True
                            else:
                                break
                        chain_pos[ci] += 1
                    else:
                        break
                    continue
                
                if v not in computed:
                    # Need to compute
                    # Check if all parents are in fast memory
                    parents_ready = True
                    for p in parents_map.get(v, []):
                        if p not in in_fast:
                            parents_ready = False
                            break
                    
                    if parents_ready:
                        if len(in_fast) < fast_memory_budget or can_evict_for_compute(v):
                            ops.extend(compute_vertex(v))
                            progress = True
                            chain_pos[ci] += 1
                        else:
                            break
                    else:
                        # Need to load some parent
                        parent_loaded = False
                        for p in parents_map.get(v, []):
                            if p not in in_fast and p in in_slow:
                                if can_fit_in_fast_or_make_room():
                                    ops.extend(load_vertex(p))
                                    parent_loaded = True
                                    progress = True
                                else:
                                    break
                        
                        if not parent_loaded:
                            break
                        continue
                
                chain_pos[ci] += 1
    
        # Check if we're making progress
        # If no progress in a full round, we may be stuck
        if rounds > 500 and not any(chain_pos[ci] < len(chains[ci]) for ci in range(len(chains))):
            break
    
    # Ensure all outputs are in fast memory
    for v in sorted(outputs):
        if v not in in_fast:
            if v in in_slow:
                # Load it
                # Make room if needed
                for u in list(in_fast):
                    # Evict a vertex that's not an output and is already stored
                    if u not in outputs and u in stored:
                        in_fast.remove(u)
                        op_count += 1
                        ops.append(['store', u])
                        break
                
                in_slow.remove(v)
                in_fast.add(v)
                op_count += 1
                ops.append(['load', v])
            elif v not in in_fast and v not in computed:
                # Shouldn't happen, but compute if needed
                # This is a fallback
                pass
    
    return ops
    
# EVOLVE-BLOCK-END
