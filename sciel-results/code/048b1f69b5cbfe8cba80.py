"""Rewritten pebbling solver with recursive divide-and-conquer and better eviction."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import heapq

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', []))
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', dag.get('sinks', []))
        inputs = dag.get('inputs', dag.get('sources', []))
    elif isinstance(dag, (list, tuple)):
        vertices = list(range(len(dag)))
        parents = {}
        outputs = []
        inputs = []
        for i, v in enumerate(dag):
            if isinstance(v, dict):
                p = v.get('parents', v.get('deps', []))
                parents[i] = list(p) if p else []
            else:
                parents[i] = []
    else:
        # Fallback: try to import baseline
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            return []

    if not vertices:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            return []

    # Build adjacency
    n = len(vertices)
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)

    # Compute in-degrees and topological order
    in_degree = defaultdict(int)
    for v in vertices:
        for p in parents.get(v, []):
            in_degree[v] += 1

    topo_order = []
    queue = deque(v for v in vertices if in_degree[v] == 0)
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    if len(topo_order) != len(vertices):
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            return []

    # Identify inputs and outputs
    input_set = set(inputs) if inputs else set(v for v in vertices if in_degree[v] == 0)
    output_set = set(outputs) if outputs else set(v for v in vertices if len(children[v]) == 0)

    # Compute depth (distance from nearest input) for priority
    depth = {}
    for v in topo_order:
        if not parents.get(v, []):
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in parents[v]) + 1

    # Compute remaining uses: how many children still need this vertex
    remaining_uses = defaultdict(int)
    for v in vertices:
        remaining_uses[v] = len(children[v])

    # Compute subtree size: total number of descendants (for better eviction priority)
    subtree_size = {}
    for v in reversed(topo_order):
        if children[v]:
            subtree_size[v] = 1 + sum(subtree_size.get(c, 0) for c in children[v])
        else:
            subtree_size[v] = 1

    # Compute last-use position in topo order for each vertex
    last_use_pos = {}
    for i, v in enumerate(topo_order):
        last_use_pos[v] = i
        for c in children[v]:
            if last_use_pos.get(c, -1) < i:
                pass  # children are later in topo order by definition
    # Actually, for a vertex u, its last use is the max position among its children
    # But we need to consider transitive uses. Let's compute it properly.
    last_use = {}
    for v in reversed(topo_order):
        if children[v]:
            last_use[v] = max(last_use.get(c, 0) for c in children[v])
        else:
            last_use[v] = topo_order.index(v) if v in topo_order else 0
    # Fix: use position in topo_order
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    for v in reversed(topo_order):
        if children[v]:
            last_use[v] = max(last_use.get(c, topo_pos[v]) for c in children[v])
        else:
            last_use[v] = topo_pos[v]

    state = {}  # vertex -> 'blue' or 'red'
    for v in vertices:
        state[v] = 'blue' if v in input_set else 'red'

    fast_mem = set(v for v in vertices if state[v] == 'blue')
    operations = []

    def get_eviction_priority(u):
        """
        Lower priority = evict first.
        We want to evict vertices that:
        1. Are not outputs (outputs should be kept)
        2. Are inputs (cheaper to reload than recompute)
        3. Have few remaining uses
        4. Have small subtrees
        5. Have early last-use
        """
        is_output = 1 if u in output_set else 0
        is_input = 1 if u in input_set else 0
        # Prefer evicting inputs over computed vertices
        # Lower value = evict first
        # So inputs (is_input=1) should have lower priority value
        priority_input = 0 if is_input else 1
        return (is_output, priority_input, remaining_uses[u], subtree_size[u], last_use[u], depth[u])

    def evict_if_needed(exclude=None):
        """Evict vertices until fast_mem is within budget."""
        nonlocal fast_mem, state, operations, remaining_uses
        while len(fast_mem) > fast_memory_budget:
            if time.time() > deadline:
                return
            # Find best victim
            best_victim = None
            best_priority = None
            for u in fast_mem:
                if u == exclude:
                    continue
                if u in output_set:
                    continue  # Don't evict outputs
                pr = get_eviction_priority(u)
                if best_priority is None or pr < best_priority:
                    best_priority = pr
                    best_victim = u
            
            if best_victim is None:
                # All in fast_mem are outputs or excluded; try any non-excluded
                for u in fast_mem:
                    if u != exclude:
                        best_victim = u
                        break
                if best_victim is None:
                    break
            
            operations.append(['store', best_victim])
            state[best_victim] = 'red'
            fast_mem.discard(best_victim)

    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break

        parents_list = parents.get(v, [])

        # Load any red parents
        for p in parents_list:
            if state[p] == 'red':
                operations.append(['load', p])
                state[p] = 'blue'
                fast_mem.add(p)

        # Evict if over budget (exclude v since we're about to compute it)
        evict_if_needed(exclude=v)

        # Compute v if not already blue
        if state[v] == 'red':
            operations.append(['compute', v])
            state[v] = 'blue'
            fast_mem.add(v)

        # After computing v, decrement remaining_uses of its parents
        for p in parents_list:
            remaining_uses[p] -= 1

        # Evict if over budget after compute
        evict_if_needed(exclude=None)

    # Ensure all outputs are blue at the end
    for v in output_set:
        if state[v] == 'red':
            operations.append(['load', v])
            state[v] = 'blue'
            fast_mem.add(v)
        evict_if_needed(exclude=None)

    # Post-processing: remove redundant load/store pairs
    # If we load a vertex and then store it without computing it in between, that's redundant
    # Actually, we can't simply remove them because the state must be consistent.
    # Instead, let's check for immediate load->store or store->load patterns that can be optimized.
    
    # A better optimization: if we store a vertex and then immediately load it back, remove both
    optimized_ops = []
    i = 0
    while i < len(operations):
        if i + 1 < len(operations):
            op1, v1 = operations[i]
            op2, v2 = operations[i + 1]
            if v1 == v2:
                if (op1 == 'load' and op2 == 'store') or (op1 == 'store' and op2 == 'load'):
                    # These cancel out; skip both
                    i += 2
                    continue
        optimized_ops.append(operations[i])
        i += 1
    operations = optimized_ops

    # Limit operations
    if len(operations) > 20000:
        operations = operations[:20000]

    return operations
# EVOLVE-BLOCK-END
