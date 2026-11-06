"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def _topo_sort(dag):
    """Return list of vertices in topological order."""
    n = len(dag)
    in_deg = [0] * n
    children = [[] for _ in range(n)]
    for v in range(n):
        parents = dag[v]
        for p in parents:
            in_deg[v] += 1
            children[p].append(v)
    queue = [v for v in range(n) if in_deg[v] == 0]
    order = []
    while queue:
        # Use stack for deterministic behavior
        v = queue.pop()
        order.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    return order

def _compute_pebbling(dag, fast_memory_budget):
    """
    Generate a pebbling schedule for a DAG.
    
    Strategy: Use a recursive pebbling approach that minimizes transfers.
    For layered DAGs, we can use a bottom-up pebbling strategy.
    
    Returns a list of [operation, vertex] pairs.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s margin
    
    n = len(dag)
    if n == 0:
        return []
    
    # Identify input vertices (no parents) and output vertices
    # In the task, inputs start blue, outputs must finish blue
    # Let's assume outputs are the last layer or vertices with no children
    
    # Build reverse adjacency
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in dag[v]:
            children[p].append(v)
    
    # Topological order
    order = _topo_sort(dag)
    pos = {v: i for i, v in enumerate(order)}
    
    # Identify inputs (no parents) and outputs (no children)
    inputs = [v for v in range(n) if len(dag[v]) == 0]
    outputs = [v for v in range(n) if len(children[v]) == 0]
    
    # Simple strategy: 
    # 1. Load all inputs
    # 2. Compute vertices in topological order, keeping results in fast memory
    # 3. When fast memory is full, store (delete) the oldest result that's no longer needed
    # 4. Reload if needed
    
    # Better strategy: Use a greedy approach that tracks which vertices are "needed"
    
    ops = []
    fast_mem = set()  # vertices currently in fast memory (red)
    blue = set(inputs)  # vertices that are blue (available in slow memory)
    
    def is_needed(v):
        """Check if vertex v is needed for any future computation."""
        # v is needed if any of its children hasn't been computed yet
        for c in children[v]:
            if c not in blue and c not in fast_mem:
                return True
        return False
    
    # Process in topological order
    computed = set()
    for v in order:
        # Check deadline
        if time.time() > deadline:
            break
        
        if v in inputs:
            # Input vertices start blue, no need to load
            continue
        
        # Check if all parents are available (blue or in fast memory)
        parents_available = all(p in blue or p in fast_mem for p in dag[v])
        
        if not parents_available:
            # Need to load some parents from slow memory
            for p in dag[v]:
                if p not in fast_mem and p not in blue:
                    # Load parent
                    ops.append(["load", p])
                    fast_mem.add(p)
                elif p in fast_mem:
                    pass
                elif p in blue:
                    # Parent is in slow memory (blue), load it
                    ops.append(["load", p])
                    fast_mem.add(p)
                    blue.discard(p)
        
        # Now compute v
        ops.append(["compute", v])
        fast_mem.add(v)
        computed.add(v)
        blue.discard(v)
        
        # Check fast memory budget
        while len(fast_mem) > fast_memory_budget:
            # Need to evict something
            # Find a vertex in fast_mem that is not needed and not an output
            evicted = False
            for u in list(fast_mem):
                if u in outputs:
                    continue
                if not is_needed(u):
                    ops.append(["delete", u])
                    fast_mem.discard(u)
                    blue.add(u)  # It's back in slow memory as blue
                    evicted = True
                    break
            if not evicted:
                # Evict the first non-output vertex
                for u in list(fast_mem):
                    if u not in outputs:
                        ops.append(["delete", u])
                        fast_mem.discard(u)
                        blue.add(u)
                        evicted = True
                        break
                if not evicted:
                    # All are outputs, this shouldn't happen if budget >= num_outputs
                    # Force evict the first one
                    if fast_mem:
                        u = next(iter(fast_mem))
                        ops.append(["delete", u])
                        fast_mem.discard(u)
                        blue.add(u)
    
    # Ensure all outputs are blue at the end
    for v in outputs:
        if v in fast_mem:
            ops.append(["delete", v])
            fast_mem.discard(v)
            blue.add(v)
        elif v not in blue:
            # Should not happen, but load it
            ops.append(["load", v])
            blue.add(v)
    
    return ops

def pebble(dag, fast_memory_budget):
    """
    Generate a pebbling schedule for a DAG.
    
    Args:
        dag: List of lists, where dag[v] is the list of parents of vertex v.
        fast_memory_budget: Maximum number of vertices that can be in fast memory.
    
    Returns:
        List of [operation, vertex] pairs.
    """
    try:
        return _compute_pebbling(dag, fast_memory_budget)
    except Exception as e:
        # Fallback: simple load-compute-delete for each vertex
        ops = []
        n = len(dag)
        for v in range(n):
            for p in dag[v]:
                ops.append(["load", p])
            ops.append(["compute", v])
            for p in dag[v]:
                ops.append(["delete", p])
        return ops
# EVOLVE-BLOCK-END
