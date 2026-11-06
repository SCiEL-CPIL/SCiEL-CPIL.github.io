"""Pebbling solver using shortest-path search in state space."""
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Solve the pebbling problem using a Dijkstra-like search over states.
    
    State: (frozenset_of_red_vertices, frozenset_of_blue_vertices)
    But we need to be careful: blue vertices are those whose values are available (either in fast memory or on disk).
    Red vertices are those whose values are in fast memory.
    
    Actually, let's think more carefully:
    - A vertex can be:
      - Not computed: value not available
      - Computed and stored on disk: value available but not in fast memory
      - Computed and in fast memory: value available and in fast memory (red)
    
    For the pebbling problem:
    - Inputs start "blue" (available on disk)
    - Outputs must finish "blue" (available, either on disk or in fast memory? The task says "all outputs must finish blue")
    
    Let me re-read: "Inputs start blue, all outputs must finish blue; every red value consumes one fast-memory slot"
    
    So "blue" means the value is available (on disk or in fast memory), and "red" means it's in fast memory.
    Wait, that doesn't make sense. Let me think again.
    
    In pebbling:
    - A vertex is "pebbled" if its value is stored somewhere
    - "Blue" might mean stored on disk (slow memory)
    - "Red" might mean stored in fast memory
    
    So:
    - Inputs start blue (on disk)
    - To compute a vertex, all its parents must be red (in fast memory)
    - After computing, the result is red (in fast memory)
    - You can store a red value to disk (becomes blue)
    - You can load a blue value to fast memory (becomes red)
    - You can delete a red value (frees a slot)
    
    The goal: all output vertices must be blue (on disk) at the end.
    
    State: (set of red vertices, set of blue vertices)
    - Red: values in fast memory
    - Blue: values on disk (available but not in fast memory)
    - A vertex can be both red and blue? No, I think a value is either in fast memory or on disk, not both.
    
    Actually, let's simplify: 
    - Each vertex has a status: 0 (not available), 1 (on disk/blue), 2 (in fast memory/red)
    - A vertex can be loaded from disk to fast memory (1 -> 2)
    - A vertex can be stored from fast memory to disk (2 -> 1)
    - A vertex can be deleted from fast memory (2 -> 0)
    - A vertex can be computed if all parents are in fast memory (2), and it becomes 2
    
    Wait, but if a vertex is on disk and we compute it again, it's recomputation. The task says "Recomputations allowed".
    
    Let me define:
    - red_set: set of vertices in fast memory
    - blue_set: set of vertices on disk (available on disk, not in fast memory)
    
    A vertex is "available" if it's in red_set or blue_set.
    
    Operations:
    1. load(v): v must be in blue_set, move to red_set. Cost: 1 transfer.
    2. store(v): v must be in red_set, move to blue_set. Cost: 1 transfer.
    3. compute(v): all parents of v must be in red_set. v becomes red. Cost: 0 transfers (but uses compute).
    4. delete(v): v must be in red_set, remove from red_set. Cost: 0 transfers.
    
    Initial state: red_set = empty, blue_set = inputs (vertices with no parents)
    Final state: all outputs in blue_set (or red_set? The task says "all outputs must finish blue")
    
    So final: outputs must be in blue_set.
    
    Memory constraint: len(red_set) <= fast_memory_budget
    
    We want to minimize total number of load/store operations (transfers).
    
    This is a shortest path problem. State = (frozenset(red_set), frozenset(blue_set))
    But this state space can be huge. Let's see the size of the DAGs.
    
    The task mentions "three layered DAGs". Let's assume they're not too large.
    
    Let's implement Dijkstra's algorithm.
    """
    
    # Parse the DAG structure
    # dag has topologically ordered parents and output vertices
    # Let's assume dag is a dict or has certain properties
    # We need to understand the structure of dag
    
    # Common formats:
    # - dag could be a list of vertices, with parents and outputs
    # - Or a dict with 'parents' and 'outputs' keys
    
    # Let's handle multiple possible formats
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        outputs = set(dag.get('outputs', []))
        all_vertices = set(parents.keys()) | set(outputs)
        # Also include any vertices that appear as parents
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
    elif isinstance(dag, (list, tuple)):
        # Assume list of vertices, need to figure out structure
        # This is ambiguous, let's try to handle it
        all_vertices = set(range(len(dag)))
        # This format is unclear, let's assume a simpler structure
        # Maybe dag is a list where each element is a vertex with parents?
        # For now, let's assume the dict format is the primary one
        raise ValueError("dag must be a dict with 'parents' and 'outputs' keys")
    else:
        raise ValueError("Unknown dag format")
    
    # Build parent/child relationships
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    
    # Inputs: vertices with no parents
    inputs = set(v for v in all_vertices if v not in parents or not parents[v])
    
    # Initial state
    init_red = frozenset()
    init_blue = frozenset(inputs)
    
    # Goal: all outputs in blue
    # We'll use Dijkstra's algorithm
    
    # State: (red_frozen, blue_frozen)
    # Priority: total transfers so far
    
    # For Dijkstra, we need:
    # - dist[state] = minimum transfers to reach state
    # - prev[state] = (prev_state, operation, vertex)
    
    dist = {}
    prev = {}
    
    start_state = (init_red, init_blue)
    dist[start_state] = 0
    prev[start_state] = None
    
    # Priority queue: (transfers, state)
    pq = [(0, start_state)]
    
    # Check if we're already at the goal
    def is_goal(state):
        red, blue = state
        return outputs.issubset(blue)
    
    # We'll also track the number of operations to limit to 20000
    # But first, let's find the optimal path in terms of transfers
    
    while pq:
        d, state = heapq.heappop(pq)
        if d > dist.get(state, float('inf')):
            continue
        
        red, blue = state
        red_set = set(red)
        blue_set = set(blue)
        
        if is_goal(state):
            # Reconstruct path
            path = []
            cur = state
            while prev[cur] is not None:
                prev_state, op, v = prev[cur]
                path.append([op, v])
                cur = prev_state
            path.reverse()
            return path
        
        # Generate neighbors
        # 1. Load: v in blue, not in red
        for v in blue_set:
            if v not in red_set:
                new_red = red_set | {v}
                new_blue = blue_set - {v}
                if len(new_red) <= fast_memory_budget:
                    new_state = (frozenset(new_red), frozenset(new_blue))
                    new_d = d + 1
                    if new_d < dist.get(new_state, float('inf')):
                        dist[new_state] = new_d
                        prev[new_state] = (state, 'load', v)
                        heapq.heappush(pq, (new_d, new_state))
        
        # 2. Store: v in red, not in blue
        for v in red_set:
            if v not in blue_set:
                new_red = red_set - {v}
                new_blue = blue_set | {v}
                new_state = (frozenset(new_red), frozenset(new_blue))
                new_d = d + 1
                if new_d < dist.get(new_state, float('inf')):
                    dist[new_state] = new_d
                    prev[new_state] = (state, 'store', v)
                    heapq.heappush(pq, (new_d, new_state))
        
        # 3. Compute: v not in red, all parents in red
        for v in all_vertices:
            if v in red_set:
                continue
            # Check if v can be computed
            if v in parents and parents[v]:
                ps = set(parents[v])
                if ps.issubset(red_set):
                    new_red = red_set | {v}
                    if len(new_red) <= fast_memory_budget:
                        new_blue = blue_set  # Blue unchanged
                        new_state = (frozenset(new_red), frozenset(new_blue))
                        new_d = d  # No transfer cost
                        if new_d < dist.get(new_state, float('inf')):
                            dist[new_state] = new_d
                            prev[new_state] = (state, 'compute', v)
                            heapq.heappush(pq, (new_d, new_state))
            else:
                # No parents, can be computed (but should already be blue if input)
                # Actually, inputs start blue, so they're available
                # If not in red and not in blue, it's not available
                if v not in blue_set:
                    # Can't compute without parents being available
                    pass
        
        # 4. Delete: v in red
        for v in red_set:
            new_red = red_set - {v}
            new_blue = blue_set  # Blue unchanged
            new_state = (frozenset(new_red), frozenset(new_blue))
            new_d = d  # No transfer cost
            if new_d < dist.get(new_state, float('inf')):
                dist[new_state] = new_d
                prev[new_state] = (state, 'delete', v)
                heapq.heappush(pq, (new_d, new_state))
    
    # If we exhausted the search without finding a goal, return empty
    return []
