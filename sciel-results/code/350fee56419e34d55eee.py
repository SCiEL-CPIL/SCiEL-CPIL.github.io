"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    dag: dict with keys:
        'vertices': list of vertex ids in topological order
        'parents': dict vertex -> list of parent vertices (topologically earlier)
        'outputs': list of output vertex ids
        'inputs': list of input vertex ids (start blue)
    fast_memory_budget: int, max red values in fast memory
    Returns: list of [op, vertex] where op in {load, store, compute, delete}
    """
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s

    # Parse DAG
    vertices = list(dag.get('vertices', []))
    parents = dag.get('parents', {})
    outputs = list(dag.get('outputs', []))
    inputs = list(dag.get('inputs', []))

    if not vertices:
        return []

    # Build children map for understanding structure
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children.setdefault(p, []).append(v)

    # Topological order is given by 'vertices'
    top_order = vertices

    # LRU fast memory
    # fast_mem: list where index 0 is most recently used
    fast_mem = []
    fast_set = set()
    blue_set = set(inputs)  # inputs start blue (available in slow memory)

    operations = []
    budget = max(0, int(fast_memory_budget))

    def is_deadline():
        return time.time() > deadline

    def touch(v):
        """Mark v as most recently used in fast memory."""
        if v in fast_set:
            fast_mem.remove(v)
            fast_mem.append(v)

    def evict_if_needed(need_slots=1):
        """Evict least recently used vertices until we have need_slots free."""
        while len(fast_mem) + need_slots > budget and fast_mem:
            v = fast_mem.pop(0)  # LRU
            fast_set.discard(v)
            operations.append(['delete', v])

    def ensure_red(v):
        """Ensure vertex v is red in fast memory. Returns True if already red."""
        if v in fast_set:
            touch(v)
            return True
        # If it's an input, load it
        if v in blue_set:
            evict_if_needed(1)
            fast_mem.append(v)
            fast_set.add(v)
            operations.append(['load', v])
            return True
        # Otherwise, it must have been computed before but evicted.
        # Recompute it: ensure its parents are red, compute, store.
        for p in parents.get(v, []):
            ensure_red(p)
        # Now compute v
        evict_if_needed(1)
        fast_mem.append(v)
        fast_set.add(v)
        operations.append(['compute', v])
        operations.append(['store', v])
        return True

    # Process vertices in topological order
    for v in top_order:
        if is_deadline():
            break
        # Ensure this vertex is red (computed and in fast memory)
        if v in fast_set:
            touch(v)
            continue
        # If it's an input, it's blue; we'll load it when needed as a parent
        # But if it's an output, it must finish blue, so we need to compute it
        # Actually, all vertices need to be computed if they're not inputs
        # Inputs are already blue; we don't need to compute them
        if v in blue_set:
            # It's an input, already blue. No action needed unless it's an output
            # that must finish blue (which it already is)
            continue
        # Compute this vertex
        for p in parents.get(v, []):
            ensure_red(p)
        # Compute v
        evict_if_needed(1)
        fast_mem.append(v)
        fast_set.add(v)
        operations.append(['compute', v])
        operations.append(['store', v])

    # Ensure all outputs are blue (they start blue if inputs, or computed)
    # Outputs that are inputs are already blue.
    # Outputs that are computed are red; we need to store them to blue.
    for v in outputs:
        if is_deadline():
            break
        if v in fast_set:
            # It's red, store to make it blue
            operations.append(['store', v])
            # Remove from fast memory since it's now blue
            if v in fast_set:
                fast_set.discard(v)
                if v in fast_mem:
                    fast_mem.remove(v)
        # If it's already blue (input), no action needed

    # If we hit deadline, return what we have
    return operations
# EVOLVE-BLOCK-END
