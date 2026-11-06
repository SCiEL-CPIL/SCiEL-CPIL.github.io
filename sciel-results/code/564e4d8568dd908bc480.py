"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling with bounded fast memory.
    """
    start_time = time.time()
    DEADLINE = 85.0

    # Parse DAG
    if isinstance(dag, dict):
        vertices = dag.get('vertices', None)
        parents = dag.get('parents', {})
        outputs = set(dag.get('outputs', []))
        inputs = dag.get('inputs', None)
    else:
        vertices = list(range(len(dag)))
        parents = {}
        outputs = set()
        inputs = None

    if vertices is None:
        all_v = set(parents.keys())
        for ps in parents.values():
            all_v.update(ps)
        vertices = sorted(all_v)

    if inputs is None:
        parent_set = set()
        for ps in parents.values():
            parent_set.update(ps)
        inputs = [v for v in vertices if v not in parent_set]

    # Build children
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Compute remaining uses: how many more times each vertex will be needed
    # Process in reverse topological order
    remaining_uses = {}
    for v in reversed(vertices):
        cnt = 1 if v in outputs else 0
        for c in children[v]:
            cnt += 1
        remaining_uses[v] = cnt

    # Greedy pebbling
    operations = []
    fast_mem = {}  # vertex -> True
    blue = set(inputs)  # inputs start blue

    # LRU with reference count awareness
    # Use a priority queue: (negative remaining_uses, timestamp, vertex)
    # Higher remaining_uses = keep longer
    import heapq
    mem_heap = []  # (-remaining_uses, seq, vertex)
    seq_counter = [0]
    in_mem = {}  # vertex -> (heap_entry_index, seq)

    def add_to_mem(v):
        """Add vertex to fast memory"""
        if len(fast_mem) >= fast_memory_budget:
            # Evict vertex with lowest remaining_uses
            while mem_heap:
                neg_ru, s, ev = heapq.heappop(mem_heap)
                if ev in fast_mem:
                    del fast_mem[ev]
                    if ev in in_mem:
                        del in_mem[ev]
                    operations.append(['delete', ev])
                    break
            if len(fast_mem) >= fast_memory_budget and not mem_heap:
                # Force evict
                for ev in list(fast_mem.keys()):
                    del fast_mem[ev]
                    if ev in in_mem:
                        del in_mem[ev]
                    operations.append(['delete', ev])
                    break

        fast_mem[v] = True
        seq_counter[0] += 1
        heapq.heappush(mem_heap, (-remaining_uses[v], seq_counter[0], v))
        in_mem[v] = (len(mem_heap) - 1, seq_counter[0])

    # Process vertices in topological order
    for v in vertices:
        if time.time() - start_time > DEADLINE:
            break

        if v in inputs:
            continue

        ps = parents.get(v, [])
        for p in ps:
            if p not in fast_mem:
                # Load p
                add_to_mem(p)
                operations.append(['load', p])
                blue.discard(p)
                # Update remaining_uses for p (decrement since we're using it now)
                # Actually, remaining_uses should be decremented when we use it
                # Let's track this properly

        # Compute v
        operations.append(['compute', v])

        # Add v to fast memory
        add_to_mem(v)

        # Decrement remaining_uses for parents (they've been used)
        for p in ps:
            remaining_uses[p] -= 1
            # Update heap entry if needed (simplified: just update the value)

        # If v is an output, store it
        if v in outputs:
            operations.append(['store', v])
            blue.add(v)
            # Remove from fast memory
            if v in fast_mem:
                del fast_mem[v]
                if v in in_mem:
                    del in_mem[v]
                remaining_uses[v] -= 1  # The store counts as a use

    # Clean up
    while fast_mem:
        for ev in list(fast_mem.keys()):
            operations.append(['delete', ev])
            del fast_mem[ev]

    return operations

# EVOLVE-BLOCK-END
