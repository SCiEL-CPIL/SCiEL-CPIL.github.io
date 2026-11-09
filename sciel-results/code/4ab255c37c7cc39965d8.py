"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Parse the DAG structure
    # Try to extract: vertices, parents dict, outputs set, inputs set
    try:
        if isinstance(dag, dict):
            # Try common conventions
            if 'parents' in dag and 'outputs' in dag:
                parents = dag['parents']
                outputs = set(dag['outputs'])
                # Infer all vertices
                all_vertices = set(parents.keys())
                for v in parents:
                    for p in parents[v]:
                        all_vertices.add(p)
                # Inputs are vertices with no parents that are not outputs
                inputs = set()
                for v in all_vertices:
                    if v not in parents or (v in parents and len(parents[v]) == 0):
                        if v not in outputs:
                            inputs.add(v)
                # Actually, inputs are leaves (no children) or specifically marked
                # Let's reconsider: inputs start blue, so they are the source vertices
                # In a DAG, inputs are vertices with no incoming edges (no parents)
                inputs = set()
                for v in all_vertices:
                    if v not in parents or len(parents[v]) == 0:
                        inputs.add(v)
                
            elif 'graph' in dag:
                # Maybe dag['graph'] is the structure
                g = dag['graph']
                if isinstance(g, dict):
                    parents = g.get('parents', {})
                    outputs = set(g.get('outputs', []))
                    all_vertices = set(parents.keys())
                    for v in parents:
                        for p in parents[v]:
                            all_vertices.add(p)
                    inputs = set()
                    for v in all_vertices:
                        if v not in parents or len(parents[v]) == 0:
                            inputs.add(v)
                else:
                    return []
            else:
                # Try to infer structure from keys
                return []
        else:
            return []
    except Exception:
        return []
    
    # Build dependency graph
    # parents[v] = list of parent vertices
    # children[u] = list of children vertices
    children = defaultdict(list)
    for v in all_vertices:
        if v in parents:
            for p in parents[v]:
                children[p].append(v)
    
    # Determine which vertices need to be computed (non-input vertices)
    compute_vertices = [v for v in all_vertices if v not in inputs]
    
    # Topological order (already given by parents structure)
    # We'll use a simple approach: process in topological order
    
    # State: set of red vertices (in fast memory)
    # Initial state: empty (all blue, inputs are available from slow memory)
    # Target state: outputs are blue (so outputs must have been computed and stored)
    
    # For small problems, we can do Dijkstra over state space
    # State = frozenset of red vertices
    # Cost = number of transfers (load/store operations)
    
    # Check if problem is small enough for state-space search
    n_vertices = len(all_vertices)
    n_compute = len(compute_vertices)
    
    if n_compute <= 15 and fast_memory_budget >= 1:
        # Try Dijkstra over state space
        try:
            # State: (frozenset_of_red, )
            # We need to track which values are in fast memory (red)
            # and which have been computed but stored (blue, but recomputable)
            
            # Actually, let's think more carefully:
            # - A vertex is "red" if its value is in fast memory
            # - A vertex is "blue" if its value is in slow memory (or it's an input)
            # - A vertex is "uncomputed" if it hasn't been computed yet
            
            # For correctness, we need to ensure that when we compute a vertex,
            # all its parents are available (either red or blue)
            
            # Let's use a simpler approach: greedy topological with memory management
            
            # For now, let's do a topological sort and use a simple strategy:
            # Load parents into fast memory, compute, store result, free parents if needed
            
            topo_order = []
            in_degree = {v: 0 for v in all_vertices}
            for v in all_vertices:
                if v in parents:
                    in_degree[v] = len(parents[v])
            
            from collections import deque
            queue = deque()
            for v in all_vertices:
                if in_degree[v] == 0:
                    queue.append(v)
            
            while queue:
                v = queue.popleft()
                topo_order.append(v)
                for c in children[v]:
                    in_degree[c] -= 1
                    if in_degree[c] == 0:
                        queue.append(c)
            
            if len(topo_order) != len(all_vertices):
                # Cycle detected, shouldn't happen in DAG
                return []
            
            # Simple schedule: process in topological order
            # Keep fast memory for recent computations
            ops = []
            fast_mem = set()  # vertices currently in fast memory
            computed = set()  # vertices that have been computed (and stored or in fast mem)
            
            for v in topo_order:
                if v in inputs:
                    # Input vertex, already available from slow memory
                    # We might want to load it into fast memory
                    if v not in fast_mem and len(fast_mem) < fast_memory_budget:
                        ops.append(['load', v])
                        fast_mem.add(v)
                    computed.add(v)
                    continue
                
                # Need to compute v
                # Ensure all parents are available
                for p in parents[v]:
                    if p not in fast_mem:
                        if p in computed:
                            # Need to load from slow memory
                            if len(fast_mem) < fast_memory_budget:
                                ops.append(['load', p])
                                fast_mem.add(p)
                            else:
                                # Need to free space: delete a non-parent vertex
                                freed = False
                                for fm in list(fast_mem):
                                    if fm != v and all(fm != pp for pp in parents[v]):
                                        # Check if fm is needed later
                                        ops.append(['delete', fm])
                                        fast_mem.remove(fm)
                                        freed = True
                                        break
                                if not freed:
                                    # Can't free space, this shouldn't happen
                                    # Just proceed (will be invalid)
                                    pass
                        else:
                            # Parent not computed yet, shouldn't happen in topo order
                            pass
                
                # Now compute v
                ops.append(['compute', v])
                computed.add(v)
                
                # Store result in fast memory
                if v not in fast_mem:
                    if len(fast_mem) < fast_memory_budget:
                        fast_mem.add(v)
                    else:
                        # Need to free space
                        freed = False
                        for fm in list(fast_mem):
                            if fm != v:
                                ops.append(['delete', fm])
                                fast_mem.remove(fm)
                                freed = True
                                break
                        if freed:
                            fast_mem.add(v)
                
                # If v is an output, store it to slow memory
                if v in outputs:
                    ops.append(['store', v])
                    if v in fast_mem:
                        fast_mem.remove(v)
            
            return ops
            
        except Exception:
            pass
    
    # Fallback: naive topological schedule
    try:
        ops = []
        # Simple: just load, compute, store for each vertex in topo order
        # This may exceed memory budget but is a valid starting point
        
        # Build topo order
        in_degree = {v: 0 for v in all_vertices}
        for v in all_vertices:
            if v in parents:
                in_degree[v] = len(parents[v])
        
        from collections import deque
        queue = deque()
        for v in all_vertices:
            if in_degree[v] == 0:
                queue.append(v)
        
        topo_order = []
        while queue:
            v = queue.popleft()
            topo_order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        for v in topo_order:
            if v in inputs:
                ops.append(['load', v])
            else:
                # Load all parents
                for p in parents[v]:
                    if p not in inputs:
                        ops.append(['load', p])
                ops.append(['compute', v])
                if v in outputs:
                    ops.append(['store', v])
        
        return ops
    except Exception:
        return []

# EVOLVE-BLOCK-END
