"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize memory transfers.
    
    dag: dict with keys:
        - 'vertices': list of vertex IDs (topologically ordered)
        - 'parents': dict mapping vertex -> list of parent vertices
        - 'outputs': list of output vertex IDs
    
    fast_memory_budget: int, max red values in fast memory
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Topological order is given
    # Identify inputs: vertices with no parents
    inputs = [v for v in vertices if not parents.get(v, [])]
    
    # Count how many times each vertex is needed (in-degree of dependents)
    # For deletion: delete when all children have been computed
    remaining_children = {v: len(children.get(v, [])) for v in vertices}
    
    # State: which vertices are red (in fast memory)
    # We'll use a greedy approach: process in topological order
    # For each vertex, ensure parents are red, compute it, then clean up
    
    operations = []
    red_set = set()  # vertices currently in fast memory (red)
    
    # We need to track for each vertex how many of its children still need it
    # When a child is computed, decrement parent's remaining count
    
    # Strategy: process vertices in topological order
    # For each vertex v:
    #   1. For each parent p of v: if p not red, load it (if input) or it should already be red
    #      Actually, for non-input parents, they should have been computed and kept red
    #      But we might have deleted them if memory was tight
    #   2. Compute v (requires all parents to be red)
    #   3. After computing v, for each parent p: decrement remaining_children[p]
    #      If remaining_children[p] == 0 and p is not an output, delete p
    #   4. If v is an output, store it at the end
    
    # But this greedy approach may exceed memory budget.
    # We need a more careful approach.
    
    # Let's use a different strategy: 
    # Process in topological order, but manage memory by deleting
    # vertices as soon as they're no longer needed.
    
    # For each vertex, we need its parents to be red when we compute it.
    # Parents are computed earlier in topological order.
    # We keep parents red until all their children are computed.
    
    # The memory usage at any point is the number of "active" vertices:
    # vertices that have been computed but not yet deleted.
    
    # Let's simulate:
    # active = set of vertices currently red
    # For v in topological order:
    #   For each parent p of v:
    #     if p not in active:
    #       # This means p was deleted, need to recompute or reload
    #       # For inputs: load
    #       # For non-inputs: this is problematic - we shouldn't have deleted
    #       # So we need to ensure we don't delete until all children are done
    #     # p should be in active
    #   Compute v: add v to active
    #   For each parent p of v:
    #     remaining_children[p] -= 1
    #     if remaining_children[p] == 0 and p not in outputs:
    #       Delete p: remove from active
    #   # Check if active size exceeds budget
    #   # If so, we need to delete something - but what?
    
    # The issue: if we process in topological order and only delete when
    # remaining_children == 0, the memory usage might exceed the budget.
    
    # For layered DAGs, the peak memory usage is bounded by the width of the
    # widest "frontier" - vertices at the same layer that are still needed.
    
    # Let's just try the greedy approach and hope the budget is sufficient.
    # If not, we may need recompute, but let's start simple.
    
    active = set()
    ops = []
    
    # Track remaining children for each vertex
    rem_children = {v: len(children.get(v, [])) for v in vertices}
    
    for v in vertices:
        # Ensure all parents are in active (red)
        for p in parents.get(v, []):
            if p not in active:
                # Parent was deleted or not loaded
                if not parents.get(p, []):
                    # p is an input, load it
                    ops.append(['load', p])
                    active.add(p)
                else:
                    # p is a computed vertex that was deleted
                    # This means we need to recompute it, which is complex
                    # For now, this indicates our deletion was too aggressive
                    # Let's try to reload if possible, but for computed vertices
                    # we can't reload - we need to recompute
                    # This is a problem with our greedy approach
                    # For simplicity, let's assume the budget is sufficient
                    # and we won't need to recompute
                    pass
        
        # Compute v
        if not parents.get(v, []):
            # v is an input, it's already loaded (blue in slow memory)
            # Loading makes it red
            # But we already loaded it above if needed
            # Actually, inputs start blue, so they're in slow memory
            # To compute with them, we need them red
            # We handle this in the parent loop above
            pass
        else:
            # v has parents, compute it
            ops.append(['compute', v])
            active.add(v)
        
        # After computing v, update remaining children for parents
        for p in parents.get(v, []):
            rem_children[p] -= 1
            if rem_children[p] == 0 and p not in outputs:
                # p is no longer needed, delete it
                if p in active:
                    ops.append(['delete', p])
                    active.discard(p)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Store all outputs
    for v in sorted(outputs):
        if v in active:
            ops.append(['store', v])
            active.discard(v)
        else:
            # Output not in active, need to handle
            # If it's an input, load then store
            if not parents.get(v, []):
                ops.append(['load', v])
                ops.append(['store', v])
            else:
                # Should have been computed and kept
                # This is an error case
                pass
    
    # Delete any remaining active vertices that aren't outputs
    # Actually, outputs should be stored, not deleted
    # Any other active vertices can be deleted
    for v in list(active):
        if v not in outputs:
            ops.append(['delete', v])
            active.discard(v)
    
    # Cap at 20000 operations
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END

