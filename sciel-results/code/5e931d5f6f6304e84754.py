"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

def _get_parents(dag):
    """Extract parent mapping from dag structure."""
    parents = {}
    if isinstance(dag, dict):
        for v, info in dag.items():
            if isinstance(info, dict) and 'parents' in info:
                parents[v] = list(info['parents'])
            elif isinstance(info, list):
                parents[v] = list(info)
            else:
                parents[v] = []
    return parents

def _get_outputs(dag):
    """Extract output vertices from dag structure."""
    if isinstance(dag, dict):
        if 'outputs' in dag:
            return list(dag['outputs'])
        # Vertices with no children are outputs
        all_vertices = set(dag.keys())
        has_child = set()
        for v, info in dag.items():
            if isinstance(info, dict) and 'parents' in info:
                for p in info['parents']:
                    has_child.add(p)
            elif isinstance(info, list):
                for p in info:
                    has_child.add(p)
        outputs = [v for v in all_vertices if v not in has_child]
        return sorted(outputs) if outputs else list(dag.keys())[-1:]
    return []

def _get_all_vertices(dag):
    """Get all vertices in the DAG."""
    if isinstance(dag, dict):
        return list(dag.keys())
    return []

def _get_inputs(dag):
    """Get input vertices (no parents)."""
    parents = _get_parents(dag)
    all_v = _get_all_vertices(dag)
    inputs = [v for v in all_v if not parents.get(v, [])]
    return inputs

def _topological_order(dag):
    """Return topological order of vertices."""
    parents = _get_parents(dag)
    all_v = _get_all_vertices(dag)
    
    # Build children map
    children = defaultdict(list)
    in_degree = {v: 0 for v in all_v}
    for v in all_v:
        for p in parents.get(v, []):
            children[p].append(v)
            in_degree[v] += 1
    
    queue = deque([v for v in all_v if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If not all vertices processed, add remaining (cycle handling)
    if len(order) < len(all_v):
        remaining = [v for v in all_v if v not in set(order)]
        order.extend(remaining)
    
    return order

def _pebble_hierarchical(dag, fast_memory_budget):
    """
    Hierarchical divide-and-conquer pebbling for layered DAGs.
    Strategy: recursively split the DAG, process sub-DAGs,
    keeping only necessary intermediate results in fast memory.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    parents = _get_parents(dag)
    all_vertices = _get_all_vertices(dag)
    outputs = _get_outputs(dag)
    inputs = _get_inputs(dag)
    topo_order = _topological_order(dag)
    
    # Build children map
    children = defaultdict(list)
    for v in all_vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Determine which vertices are "essential" - on paths to outputs
    essential = set()
    def mark_essential(v):
        if v in essential:
            return
        essential.add(v)
        for c in children[v]:
            mark_essential(c)
    for o in outputs:
        mark_essential(o)
    
    # Count how many times each vertex's value is needed
    use_count = defaultdict(int)
    def count_uses(v):
        if v in use_count:
            return
        for c in children[v]:
            use_count[v] += 1
            count_uses(c)
    for o in outputs:
        count_uses(o)
    
    # Simple but effective strategy: 
    # 1. Load all inputs
    # 2. Process in topological order, computing each vertex
    # 3. Keep values that are needed multiple times in fast memory
    # 4. Store outputs at the end
    
    operations = []
    in_fast = set()  # vertices currently in fast memory
    fast_count = 0
    
    # Identify which non-input vertices need to stay in fast memory
    # (those with use_count > 1 and fit in budget)
    multi_use = {v for v in all_vertices if use_count[v] > 1 and v not in inputs}
    
    # Try to keep multi-use vertices in fast memory if budget allows
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        # If v is an input, load it
        if v in inputs:
            if v not in in_fast:
                operations.append(['load', v])
                in_fast.add(v)
                fast_count += 1
        else:
            # Compute v
            # Ensure all parents are in fast memory
            for p in parents.get(v, []):
                if p not in in_fast:
                    # Load parent from slow memory
                    operations.append(['load', p])
                    in_fast.add(p)
                    fast_count += 1
            
            # Check if we have room for v's result
            if fast_count >= fast_memory_budget:
                # Need to evict something
                # Evict a vertex that won't be needed again
                evictable = None
                for u in list(in_fast):
                    if u not in inputs and u != v:
                        # Check if u is still needed
                        if u not in multi_use or use_count[u] == 0:
                            evictable = u
                            break
                if evictable is None:
                    # Evict any non-input, non-output vertex
                    for u in list(in_fast):
                        if u not in inputs and u not in outputs:
                            evictable = u
                            break
                if evictable is not None:
                    operations.append(['delete', evictable])
                    in_fast.discard(evictable)
                    fast_count -= 1
            
            operations.append(['compute', v])
            in_fast.add(v)
            fast_count += 1
            
            # If v is a multi-use vertex, keep it
            # Otherwise, we can delete it after its last use
            # For now, keep it and manage later
    
    # Store all outputs
    for o in outputs:
        if time.time() > deadline:
            break
        if o in in_fast:
            operations.append(['store', o])
        else:
            # Load output first if not in fast memory
            operations.append(['load', o])
            in_fast.add(o)
            operations.append(['store', o])
    
    # Validate and clean up
    if len(operations) > 20000:
        # Truncate - this shouldn't happen for reasonable DAGs
        operations = operations[:20000]
    
    return operations

def _validate_operations(dag, operations, fast_memory_budget):
    """Validate that operations are legal and produce correct final state."""
    parents = _get_parents(dag)
    all_vertices = set(_get_all_vertices(dag))
    outputs = set(_get_outputs(dag))
    inputs = set(_get_inputs(dag))
    
    # State: 'blue' (in fast memory), 'red' (in slow memory), 'none'
    state = {}
    for v in all_vertices:
        if v in inputs:
            state[v] = 'blue'  # inputs start blue
        else:
            state[v] = 'none'
    
    fast_count = len([v for v in all_vertices if state[v] == 'blue'])
    
    for i, (op, v) in enumerate(operations):
        if v not in all_vertices:
            return False, f"Unknown vertex {v} at op {i}"
        
        if op == 'load':
            # Can only load if value exists in slow memory (red or was stored)
            # For inputs, they're always available
            if v in inputs:
                if state[v] == 'blue':
                    continue  # Already loaded, skip
                state[v] = 'blue'
                fast_count += 1
            else:
                if state[v] == 'none':
                    return False, f"Cannot load {v} at op {i}, value not computed"
                if state[v] == 'blue':
                    continue
                state[v] = 'blue'
                fast_count += 1
        elif op == 'store':
            if state[v] != 'blue':
                return False, f"Cannot store {v} at op {i}, not in fast memory"
            state[v] = 'red'
            fast_count -= 1
        elif op == 'compute':
            # All parents must be blue
            for p in parents.get(v, []):
                if state.get(p) != 'blue':
                    return False, f"Cannot compute {v} at op {i}, parent {p} not blue"
            # Result goes to fast memory
            if state[v] == 'blue':
                return False, f"Cannot compute {v} at op {i}, already blue"
            state[v] = 'blue'
            fast_count += 1
        elif op == 'delete':
            if state[v] != 'blue':
                return False, f"Cannot delete {v} at op {i}, not blue"
            state[v] = 'none'
            fast_count -= 1
        else:
            return False, f"Unknown operation {op} at op {i}"
        
        if fast_count > fast_memory_budget:
            return False, f"Fast memory overflow at op {i}: {fast_count} > {fast_memory_budget}"
    
    # All outputs must be blue at the end
    for o in outputs:
        if state.get(o) != 'blue':
            return False, f"Output {o} is not blue at end"
    
    return True, "Valid"

def _count_transfers(operations):
    """Count the number of load/store operations (transfers)."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

def _naive_schedule_transfers(dag, fast_memory_budget):
    """
    Compute the transfer count for a naive topological schedule.
    Naive: load each input once, compute each vertex once, store each output once.
    But parents may need to be reloaded if they're not kept in fast memory.
    """
    parents = _get_parents(dag)
    all_vertices = _get_all_vertices(dag)
    outputs = _get_outputs(dag)
    inputs = _get_inputs(dag)
    topo_order = _topological_order(dag)
    
    # Naive strategy: process in topological order
    # Load inputs, compute, and for each vertex, ensure parents are loaded
    # Then delete parents that won't be needed again
    
    children = defaultdict(list)
    for v in all_vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Count last use of each vertex
    last_use = {}
    for i, v in enumerate(topo_order):
        last_use[v] = i
    
    # Naive: load each vertex when first needed, delete when no longer needed
    # Transfers = loads + stores
    loads = 0
    stores = 0
    
    in_fast = set()
    fast_count = 0
    
    for v in topo_order:
        if v in inputs:
            if v not in in_fast:
                loads += 1
                in_fast.add(v)
                fast_count += 1
        else:
            # Ensure parents are in fast memory
            for p in parents.get(v, []):
                if p not in in_fast:
                    loads += 1
                    in_fast.add(p)
                    fast_count += 1
            
            # Compute v
            in_fast.add(v)
            fast_count += 1
            
            # Delete parents that are no longer needed
            for p in parents.get(v, []):
                # Check if p is needed by any other vertex after v
                needed = False
                for c in children[p]:
                    if c != v and topo_order.index(c) > topo_order.index(v) if c in topo_order else False:
                        needed = True
                        break
                if not needed and p not in outputs:
                    if p in in_fast:
                        in_fast.discard(p)
                        fast_count -= 1
    
    # Store outputs
    for o in outputs:
        stores += 1
    
    return loads + stores

def _improved_pebbling(dag, fast_memory_budget):
    """
    Improved pebbling with better memory management.
    Uses reference counting to decide when to evict.
    """
    parents = _get_parents(dag)
    all_vertices = _get_all_vertices(dag)
    outputs = set(_get_outputs(dag))
    inputs = set(_get_inputs(dag))
    topo_order = _topological_order(dag)
    
    children = defaultdict(list)
    for v in all_vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # For each vertex, compute the set of descendants that need it
    # and the position of the last descendant in topological order
    last_needed = {}
    def compute_last_needed(v, memo={}):
        if v in memo:
            return memo[v]
        if not children[v]:
            memo[v] = topo_order.index(v) if v in topo_order else 0
            return memo[v]
        max_pos = topo_order.index(v) if v in topo_order else 0
        for c in children[v]:
            pos = compute_last_needed(c)
            max_pos = max(max_pos, pos)
        memo[v] = max_pos
        return max_pos
    
    for v in all_vertices:
        compute_last_needed(v)
    
    operations = []
    in_fast = set()
    fast_count = 0
    
    for v in topo_order:
        # Load inputs
        if v in inputs:
            if v not in in_fast:
                operations.append(['load', v])
                in_fast.add(v)
                fast_count += 1
        else:
            # Ensure all parents are in fast memory
            for p in parents.get(v, []):
                if p not in in_fast:
                    # Need to load p
                    # First, make room if needed
                    if fast_count >= fast_memory_budget:
                        # Evict a vertex whose last use is before current position
                        evicted = False
                        for u in list(in_fast):
                            if u not in inputs and u not in outputs:
                                if last_needed[u] < topo_order.index(v):
                                    operations.append(['delete', u])
                                    in_fast.discard(u)
                                    fast_count -= 1
                                    evicted = True
                                    break
                        if not evicted and fast_count >= fast_memory_budget:
                            # Evict any non-critical vertex
                            for u in list(in_fast):
                                if u not in inputs and u not in outputs:
                                    operations.append(['delete', u])
                                    in_fast.discard(u)
                                    fast_count -= 1
                                    break
                    
                    operations.append(['load', p])
                    in_fast.add(p)
                    fast_count += 1
            
            # Compute v
            if fast_count >= fast_memory_budget:
                # Make room for v
                for u in list(in_fast):
                    if u not in inputs and u not in outputs:
                        if last_needed[u] < topo_order.index(v):
                            operations.append(['delete', u])
                            in_fast.discard(u)
                            fast_count -= 1
                            break
            
            operations.append(['compute', v])
            in_fast.add(v)
            fast_count += 1
            
            # Delete parents that are no longer needed
            for p in parents.get(v, []):
                if p in in_fast and p not in inputs and p not in outputs:
                    if last_needed[p] <= topo_order.index(v):
                        # Check if p is needed by any child after v
                        still_needed = False
                        for c in children[p]:
                            if c != v and c in topo_order and topo_order.index(c) > topo_order.index(v):
                                still_needed = True
                                break
                        if not still_needed:
                            operations.append(['delete', p])
                            in_fast.discard(p)
                            fast_count -= 1
    
    # Store all outputs
    for o in outputs:
        if o in in_fast:
            operations.append(['store', o])
        else:
            # Need to load o first
            if fast_count >= fast_memory_budget:
                for u in list(in_fast):
                    if u not in inputs and u not in outputs:
                        operations.append(['delete', u])
                        in_fast.discard(u)
                        fast_count -= 1
                        break
            operations.append(['load', o])
            in_fast.add(o)
            operations.append(['store', o])
    
    return operations

def pebble(dag, fast_memory_budget):
    """
    Main entry point: generate pebbling operations for a DAG.
    Returns list of [operation, vertex] pairs.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Try the improved pebbling first
    try:
        operations = _improved_pebbling(dag, fast_memory_budget)
        valid, msg = _validate_operations(dag, operations, fast_memory_budget)
        if valid and len(operations) <= 20000:
            return operations
    except Exception:
        pass
    
    # Fallback to hierarchical
    try:
        operations = _pebble_hierarchical(dag, fast_memory_budget)
        valid, msg = _validate_operations(dag, operations, fast_memory_budget)
        if valid and len(operations) <= 20000:
            return operations
    except Exception:
        pass
    
    # Last resort: simple topological with basic memory management
    parents = _get_parents(dag)
    all_vertices = _get_all_vertices(dag)
    outputs = _get_outputs(dag)
    inputs = _get_inputs(dag)
    topo_order = _topological_order(dag)
    
    operations = []
    in_fast = set()
    fast_count = 0
    
    for v in topo_order:
        if time.time() > deadline:
            break
        if v in inputs:
            if v not in in_fast:
                operations.append(['load', v])
                in_fast.add(v)
                fast_count += 1
        else:
            for p in parents.get(v, []):
                if p not in in_fast:
                    if fast_count >= fast_memory_budget:
                        # Evict something
                        for u in list(in_fast):
                            if u not in inputs and u not in outputs:
                                operations.append(['delete', u])
                                in_fast.discard(u)
                                fast_count -= 1
                                break
                    operations.append(['load', p])
                    in_fast.add(p)
                    fast_count += 1
            operations.append(['compute', v])
            in_fast.add(v)
            fast_count += 1
    
    for o in outputs:
        if time.time() > deadline:
            break
        if o not in in_fast:
            if fast_count >= fast_memory_budget:
                for u in list(in_fast):
                    if u not in inputs and u not in outputs:
                        operations.append(['delete', u])
                        in_fast.discard(u)
                        fast_count -= 1
                        break
            operations.append(['load', o])
            in_fast.add(o)
        operations.append(['store', o])
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
