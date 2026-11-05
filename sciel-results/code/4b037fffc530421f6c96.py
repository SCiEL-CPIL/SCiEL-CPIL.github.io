"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import heapq
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem using a state-space search approach.
    
    The DAG is assumed to have:
    - 'vertices': list of vertex IDs
    - 'parents': dict mapping vertex -> list of parent vertex IDs (topologically ordered)
    - 'outputs': list of output vertex IDs
    - 'inputs': list of input vertex IDs (start blue)
    
    Operations: load, store, compute, delete
    - load: bring an input vertex into fast memory (counts as 1 transfer)
    - store: move a computed value from fast to slow memory (counts as 1 transfer)
    - compute: compute a vertex from its parents in fast memory (no transfer)
    - delete: remove a vertex from fast memory (no transfer)
    
    All inputs start blue (in slow memory). All outputs must finish blue (in slow memory).
    Red values consume fast memory slots.
    
    We use Dijkstra's algorithm on the state space where:
    - State: frozenset of vertices currently in fast memory (red)
    - We track which vertices have been computed and stored to slow memory (blue)
    - Goal: all output vertices are blue (stored to slow memory)
    
    Since the state space can be large, we use A* with an admissible heuristic.
    """
    # Parse the DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
    else:
        # Try to infer structure
        vertices = list(range(len(dag)))
        parents = {}
        outputs = []
        inputs = []
        for i, v in enumerate(dag):
            if isinstance(v, dict):
                vertices[i] = v.get('id', i)
                parents[v.get('id', i)] = v.get('parents', [])
                if v.get('is_output', False):
                    outputs.append(v.get('id', i))
                if v.get('is_input', False):
                    inputs.append(v.get('id', i))
            elif isinstance(v, tuple) or isinstance(v, list):
                if len(v) >= 2:
                    vid = v[0]
                    par = v[1] if len(v) > 1 else []
                    vertices[i] = vid
                    parents[vid] = par
    
    # Ensure all vertices are known
    all_vertices = set(vertices)
    for v in parents:
        all_vertices.add(v)
        for p in parents[v]:
            all_vertices.add(p)
    all_vertices = list(all_vertices)
    vertex_set = set(all_vertices)
    
    # Determine inputs and outputs if not explicitly given
    if not inputs:
        inputs = [v for v in all_vertices if v not in parents or not parents.get(v, [])]
    if not outputs:
        # Outputs are vertices that are not parents of any other vertex
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        outputs = [v for v in all_vertices if not children.get(v, [])]
    
    inputs = list(set(inputs))
    outputs = list(set(outputs))
    output_set = set(outputs)
    input_set = set(inputs)
    
    # Topological order
    in_degree = {v: 0 for v in all_vertices}
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            if p in in_degree:
                in_degree[v] = in_degree.get(v, 0) + 1
            children[p].append(v)
    
    topo_order = []
    queue = deque([v for v in all_vertices if in_degree.get(v, 0) == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] = in_degree.get(c, 0) - 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # State: (frozenset of red vertices in fast memory, frozenset of blue vertices in slow memory)
    # Initial state: no red vertices, all inputs are blue
    initial_red = frozenset()
    initial_blue = frozenset(inputs)
    initial_state = (initial_red, initial_blue)
    
    # Goal: all outputs are blue
    def is_goal(state):
        _, blue = state
        return output_set.issubset(blue)
    
    # Dijkstra's algorithm
    # dist[state] = minimum cost to reach state
    # Cost = number of transfers (load + store operations)
    
    # Heuristic: admissible lower bound on remaining transfers
    # Each output that is not yet blue needs at least 1 store operation
    # Additionally, if fast memory is full and we need to compute more, we may need deletes
    def heuristic(state):
        red, blue = state
        remaining_outputs = output_set - blue
        # At minimum, each remaining output needs a store
        cost = len(remaining_outputs)
        # If fast memory is full, we might need to free space
        # This is a weak but admissible heuristic
        return cost
    
    # Priority queue: (f, g, state)
    # g = cost so far (transfers)
    # f = g + h
    
    start_g = 0
    start_f = heuristic(initial_state)
    
    pq = [(start_f, start_g, initial_state)]
    best_g = {initial_state: 0}
    parent = {initial_state: None}  # (prev_state, operation, vertex)
    
    max_ops = 20000
    op_count = 0
    
    while pq and op_count < max_ops:
        f, g, state = heapq.heappop(pq)
        
        if state not in best_g or g > best_g[state]:
            continue
        
        if is_goal(state):
            # Reconstruct path
            path = []
            cur = state
            while parent[cur] is not None:
                prev_state, op, vertex = parent[cur]
                path.append([op, vertex])
                cur = prev_state
            path.reverse()
            return path
        
        red, blue = state
        
        # Generate neighbors
        # 1. Load an input vertex into fast memory (if not already red)
        for v in inputs:
            if v not in red and v in input_set:
                new_red = frozenset(red | {v})
                new_blue = blue
                if len(new_red) <= fast_memory_budget:
                    new_state = (new_red, new_blue)
                    new_g = g + 1  # load is a transfer
                    if new_state not in best_g or new_g < best_g[new_state]:
                        best_g[new_state] = new_g
                        parent[new_state] = (state, 'load', v)
                        new_f = new_g + heuristic(new_state)
                        heapq.heappush(pq, (new_f, new_g, new_state))
        
        # 2. Store a red vertex to slow memory (make it blue)
        for v in red:
            new_red = frozenset(red - {v})
            new_blue = frozenset(blue | {v})
            new_state = (new_red, new_blue)
            new_g = g + 1  # store is a transfer
            if new_state not in best_g or new_g < best_g[new_state]:
                best_g[new_state] = new_g
                parent[new_state] = (state, 'store', v)
                new_f = new_g + heuristic(new_state)
                heapq.heappush(pq, (new_f, new_g, new_state))
        
        # 3. Compute a vertex whose parents are all in fast memory (red)
        for v in all_vertices:
            if v in red:
                continue
            ps = parents.get(v, [])
            if not ps:
                # Input vertex - can only be loaded, not computed
                continue
            if all(p in red for p in ps):
                if len(red) < fast_memory_budget:
                    new_red = frozenset(red | {v})
                    new_blue = blue
                    new_state = (new_red, new_blue)
                    new_g = g  # compute is not a transfer
                    if new_state not in best_g or new_g < best_g[new_state]:
                        best_g[new_state] = new_g
                        parent[new_state] = (state, 'compute', v)
                        new_f = new_g + heuristic(new_state)
                        heapq.heappush(pq, (new_f, new_g, new_state))
        
        # 4. Delete a red vertex (free up space, but lose the value)
        for v in red:
            new_red = frozenset(red - {v})
            new_blue = blue
            new_state = (new_red, new_blue)
            new_g = g  # delete is not a transfer
            if new_state not in best_g or new_g < best_g[new_state]:
                best_g[new_state] = new_g
                parent[new_state] = (state, 'delete', v)
                new_f = new_g + heuristic(new_state)
                heapq.heappush(pq, (new_f, new_g, new_state))
        
        op_count += 1
    
    # If search didn't find a solution, fall back to a simple baseline
    # Simple baseline: process in topological order, loading/storing as needed
    ops = []
    red = set()
    blue = set(inputs)
    
    for v in topo_order:
        if v in input_set:
            # Load input
            if v not in red:
                if len(red) >= fast_memory_budget:
                    # Need to free space
                    for r in list(red):
                        if r not in output_set or r in blue:
                            red.remove(r)
                            ops.append(['delete', r])
                        if len(red) < fast_memory_budget:
                            break
                red.add(v)
                ops.append(['load', v])
            continue
        
        ps = parents.get(v, [])
        if not ps:
            continue
        
        # Ensure parents are in fast memory
        for p in ps:
            if p not in red:
                if p in blue:
                    # Need to load from slow memory? No, load is only for inputs
                    # We need to recompute or have stored it
                    # For simplicity, this baseline may not work perfectly
                    pass
                else:
                    # Recompute parent
                    pass
        
        # This simple fallback is incomplete. Let's use a different approach.
    
    # Alternative simple baseline: compute all in topological order with minimal memory
    # For each vertex, compute it when its parents are available, then store it
    ops = []
    red = set()
    blue = set(inputs)
    
    # Process in topological order
    for v in topo_order:
        if v in input_set:
            if v not in red:
                if len(red) >= fast_memory_budget:
                    # Free space by deleting non-essential red vertices
                    for r in sorted(red, key=lambda x: 0):
                        red.discard(r)
                        ops.append(['delete', r])
                        if len(red) < fast_memory_budget:
                            break
                red.add(v)
                ops.append(['load', v])
            continue
        
        ps = parents.get(v, [])
        if not ps:
            continue
        
        # Check if all parents are in red
        missing = [p for p in ps if p not in red]
        if missing:
            # Need to recompute missing parents
            # This gets complex; for now, just try to load them if inputs
            for p in missing:
                if p in input_set and p not in red:
                    if len(red) >= fast_memory_budget:
                        for r in list(red):
                            red.discard(r)
                            ops.append(['delete', r])
                            if len(red) < fast_memory_budget:
                                break
                    red.add(p)
                    ops.append(['load', p])
        
        # Compute v
        if all(p in red for p in ps):
            if len(red) >= fast_memory_budget:
                # Free space
                for r in list(red):
                    if r != v:
                        red.discard(r)
                        ops.append(['delete', r])
                        if len(red) < fast_memory_budget:
                            break
            red.add(v)
            ops.append(['compute', v])
        
        # Store v if it's an output or if we need to free space
        if v in output_set:
            red.discard(v)
            blue.add(v)
            ops.append(['store', v])
    
    # Ensure all outputs are blue
    for v in outputs:
        if v not in blue:
            if v in red:
                red.discard(v)
                blue.add(v)
                ops.append(['store', v])
    
    return ops
# EVOLVE-BLOCK-END
