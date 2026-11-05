import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Minimizes communication (transfers) for DAG pebbling.
    dag: dict with 'nodes', 'edges' or similar structure. 
         Assuming standard format: 
         dag['nodes']: list of node ids
         dag['edges']: list of (parent, child)
         dag['outputs']: list of output node ids
         dag['inputs']: list of input node ids (or inferred as in-degree 0)
    fast_memory_budget: int
    Returns: list of [operation, vertex]
    """
    start_time = time.time()
    deadline = 60.0 # Safety margin before 90s kill
    
    # Parse DAG
    nodes = dag.get('nodes', [])
    edges = dag.get('edges', [])
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    if not nodes:
        return []
        
    # If inputs/outputs not explicitly given, infer
    if not inputs:
        children_count = defaultdict(int)
        for u, v in edges:
            children_count[v] += 1
        inputs = {n for n in nodes if children_count[n] == 0}
        
    if not outputs:
        parents_count = defaultdict(int)
        for u, v in edges:
            parents_count[u] += 1
        outputs = {n for n in nodes if parents_count[n] == 0}

    # Build adjacency lists
    children = defaultdict(list)
    parents = defaultdict(list)
    for u, v in edges:
        children[u].append(v)
        parents[v].append(u)
        
    # Identify relevant subgraph: ancestors of outputs
    relevant = set()
    stack = list(outputs)
    while stack:
        n = stack.pop()
        if n in relevant:
            continue
        relevant.add(n)
        for p in parents[n]:
            if p not in relevant:
                stack.append(p)
                
    # Filter nodes and edges to relevant subgraph
    # Note: Inputs might be outside relevant if they don't lead to outputs? 
    # But usually all inputs lead to outputs. If not, we ignore them.
    
    # State
    # fast: set of nodes currently in fast memory (Red)
    # slow: set of nodes currently in slow memory (Blue)
    # computed: set of nodes that have been computed at least once
    # remaining_children: for each node, how many of its children are NOT yet computed
    # This helps determine "demand" for eviction.
    
    fast = set()
    slow = set(inputs & relevant) # Only inputs that are relevant
    computed = set()
    
    # Initialize remaining children counts for relevant nodes
    # A node's "demand" is the number of children that still need it.
    # When a child is computed, the parent's demand decreases.
    # Initially, demand(n) = len(children[n] intersect relevant)
    demand = {}
    for n in relevant:
        demand[n] = len([c for c in children[n] if c in relevant])
        
    operations = []
    
    # Helper to add operation
    def add_op(op, node):
        operations.append([op, node])
        
    # Check if node is computable: all parents in fast
    def is_computable(n):
        if n in computed:
            return False # We only compute once? 
            # Wait, recomputation is allowed. 
            # But if it's in fast, we don't need to compute.
            # If it's in slow, we can load.
            # If it's not in fast/slow, we must recompute.
            # Actually, if n is in computed, its value exists in slow (if stored) or was deleted.
            # If we deleted it, we must recompute.
            # Let's track if value is available in slow.
            # If n is in slow, we can load it.
            # If n is not in slow and not in fast, we must recompute.
            pass
            
    # Let's refine the state machine.
    # State of a node:
    # 1. In Fast (Red)
    # 2. In Slow (Blue)
    # 3. Not available (Must recompute)
    
    # We can simplify:
    # If node is in `fast`, it's available for compute.
    # If node is in `slow`, we can `load` it to make it available.
    # If node is in neither, we must `compute` it from parents (which must be in fast).
    
    # However, if we `compute` a node that is already in `slow`, we are recomputing. 
    # It's better to `load` if it's in `slow`.
    
    # Revised Logic:
    # To compute node v:
    #   For each parent u:
    #     If u not in fast:
    #       If u in slow: load(u)
    #       Else: recompute(u) -> which requires its parents in fast...
    #     This is recursive.
    
    # Greedy Approach:
    # We maintain a set of "ready" nodes: nodes that are in `fast` and whose children might need them?
    # No, we want to compute children.
    # A node v is "ready to compute" if all its parents are in `fast`.
    
    # We process nodes.
    # We need to ensure we don't get stuck.
    
    # Let's use a priority queue for "candidates to compute".
    # A node v is a candidate if:
    #   1. v is in relevant
    #   2. v is not in fast (if it is, it's already there)
    #   3. All parents of v are in fast? No, if parents are not in fast, we can't compute v yet.
    #      So v is "blocked" if any parent is not in fast.
    
    # We need to unblock nodes by loading/computing parents.
    
    # This is effectively a topological sort with memory constraints.
    
    # Algorithm:
    # 1. Start with inputs in slow.
    # 2. While there are relevant nodes not in slow (and not outputs already stored?):
    #    Actually, we just need to ensure all outputs end up in slow.
    #    And we want to minimize loads/stores.
    
    # Let's try a different perspective:
    # We simulate the execution.
    # We have a set `unprocessed` = relevant nodes.
    # We have `fast` and `slow`.
    
    # While `unprocessed` is not empty:
    #   Find all nodes in `unprocessed` that are "computable":
    #     - All parents are in `fast`.
    #   If any are computable:
    #     Pick one (heuristic: highest demand? Output?)
    #     Compute it. Add to `fast`. Remove from `unprocessed`.
    #     If it's an Output, we might Store it immediately to free space?
    #        If we store it, it moves to `slow`. We remove from `fast`.
    #        This is good if space is tight.
    #   If none are computable:
    #     We need to make a parent available in `fast`.
    #     Pick a node v in `unprocessed` that is "most urgent" (e.g., closest to output).
    #     Pick a parent u of v that is NOT in `fast`.
    #     If u is in `slow`: Load u.
    #     If u is NOT in `slow`: We must recompute u.
    #        This implies u's parents must be in fast... recursion.
    #        To avoid recursion complexity in the main loop, we can just "force" the computation of u by treating it as the new target.
    #        But if u is not in slow, it means we deleted it or never computed it.
    #        If we never computed it, it's an ancestor of an input? No, inputs are in slow.
    #        So u must have been computed and deleted/stored.
    #        If stored, it's in slow. So if not in slow, it was deleted.
    #        So we must recompute u.
    #        To recompute u, we need its parents in fast.
    #        This is the same problem.
        
    # So, the "blocked" state is resolved by loading/computing parents.
    # If a parent is not in fast, we focus on that parent.
    
    # This suggests a recursive DFS pebbling algorithm is more natural.
    
    # Let's implement a recursive pebbling with memoization of "cost to pebble a subtree"?
    # No, the memory budget is global.
    
    # Given the 90s limit and likely small DAGs, a simple iterative greedy with backtracking (if stuck) might work.
    # But "stuck" shouldn't happen if we always load/compute a parent of a blocked node.
    
    # Let's refine the "Pick a parent" step.
    # If no node is computable, pick a node v in `unprocessed` with the highest priority.
    # Priority: 
    #   - Is it an output?
    #   - How many uncomputed children does it have?
    # Pick a parent u of v not in `fast`.
    # If `fast` is full, evict a node w.
    #   Eviction heuristic: 
    #     - If w is an output and not yet stored, store it? No, if we evict it, we lose it.
    #     - We should evict the node with the lowest "future use".
    #     - Future use = number of children in `unprocessed`.
    #     - If w is an output, its future use is 0 (it's a sink). So evict outputs first?
    #     - If we evict an output, we must store it (if we want it in slow) or delete it (if we recompute later).
    #     - But outputs must finish in slow. So if we evict an output, we should `store` it.
    #     - If we store it, it goes to `slow`. We can `load` it back if needed? No, outputs are sinks.
    #     - So if we evict an output, we `store` it and remove from `unprocessed`? 
    #     - Yes, once an output is stored, it's done.
    
    # So, if we evict an output, we `store` it and mark it done.
    # If we evict a non-output, we `delete` it.
    
    # Then we `load` u (if in slow) or `compute` u (if not in slow? No, if not in slow, we must recompute).
    # If u is not in slow, we must recompute u.
    # To recompute u, we need u's parents in fast.
    # This effectively changes our target to u.
    
    # This looks like a worklist algorithm.
    
    # Let's implement this.
    
    unprocessed = set(relevant)
    # We need to track which nodes are "done" (stored if output, or no longer needed).
    # Actually, if a non-output node is deleted, it's not "done", it's just "not in memory".
    # It must be recomputed if needed again.
    
    # To avoid infinite loops, we need a good heuristic.
    
    # Let's try a simpler approach: 
    # Topological sort, but with a sliding window of fast memory.
    # This is hard to optimize for minimal transfers.
    
    # Given the complexity, I will implement the Greedy Eviction approach carefully.
    
    def get_priority(node):
        # Higher priority = more important to compute/keep
        # Outputs are high priority to finish.
        # Nodes with many children are high priority to keep in fast (reuse).
        # Nodes with few parents are easy to compute?
        if node in outputs:
            return (10000, demand[node])
        return (0, demand[node])

    # We need to handle the case where we need to recompute a node not in slow.
    # This happens if we deleted it.
    # We should try to avoid deleting nodes that are likely to be needed again.
    # The `demand` heuristic helps with this.
    
    while unprocessed:
        if time.time() > deadline:
            # Emergency exit: return best effort
            # Ensure outputs are stored? 
            # If we run out of time, we might fail.
            # But for now, break.
            break
            
        # Find computable nodes
        computable = []
        for n in unprocessed:
            if all(p in fast for p in parents[n]):
                computable.append(n)
                
        if computable:
            # Pick best computable node
            # Sort by priority
            computable.sort(key=get_priority, reverse=True)
            v = computable[0]
            # Compute v
            add_op("compute", v)
            fast.add(v)
            unprocessed.remove(v)
            # Update demand of parents
            for p in parents[v]:
                demand[p] -= 1
                
            # If v is an output, should we store it now?
            # If fast is tight, yes.
            # If fast is not tight, maybe keep it?
            # But outputs don't need to be in fast for other computations.
            # So storing it immediately frees space.
            # However, if we store it, we incur a transfer cost.
            # If we keep it in fast, we avoid the store transfer now, but we occupy space.
            # Since we must store it eventually, and it doesn't help compute anything else,
            # it's generally better to store it immediately to free space, UNLESS
            # we are very close to finishing and can afford to keep it.
            # But "mean transfers" penalizes every store.
            # Wait, if we store it, we pay 1 transfer.
            # If we keep it in fast, we pay 0 now, but we might have to evict something else later,
            # which might cause a recompute (which is more expensive in terms of transfers for ancestors).
            # So storing outputs early is usually good for space, but costs 1 transfer.
            # Not storing costs 0 now, but risks eviction.
            # If we evict an output, we must store it anyway (to satisfy "finish blue").
            # So storing it now vs later costs the same 1 transfer.
            # So we should store it now to free space!
            if v in outputs:
                add_op("store", v)
                fast.remove(v)
                slow.add(v)
                # It's done.
                
        else:
            # No computable nodes. Need to load/compute a parent.
            # Pick a node v in unprocessed that is "most blocked" or high priority.
            # Let's pick the one with highest priority.
            unprocessed_list = list(unprocessed)
            unprocessed_list.sort(key=get_priority, reverse=True)
            v = unprocessed_list[0]
            
            # Find a parent u of v not in fast
            u = None
            for p in parents[v]:
                if p not in fast:
                    u = p
                    break
                    
            if u is None:
                # This shouldn't happen if v is not computable
                # But if v has no parents? It's an input.
                # Inputs are in slow.
                # If v is an input, it should be in slow.
                # We can load it.
                u = v # Treat self as parent to load? No.
                # If v is an input, parents[v] is empty.
                # So the loop finds no u.
                # We need to load v itself? No, v is the node we want to compute?
                # Inputs are not computed, they are loaded.
                # So if v is an input, we should load it.
                if v in inputs:
                    u = v
                    # Special case: load v
                    if fast_memory_budget > len(fast):
                        add_op("load", v)
                        fast.add(v)
                        slow.remove(v)
                        # v is now in fast. It is "computed" (available).
                        unprocessed.remove(v)
                        # Update demand of children? No, inputs don't have parents.
                        # But v is now available.
                        continue
                    else:
                        # Need to evict
                        # Evict logic below
                        pass
                else:
                    # Should not happen
                    continue
            
            # If u is a parent, we need u in fast.
            if u in fast:
                continue # Should not happen
            
            # We need to make u available in fast.
            # If u is in slow, load it.
            # If u is not in slow, recompute it.
            
            # Check space
            if len(fast) >= fast_memory_budget:
                # Evict a node w from fast
                # Heuristic: lowest demand
                # If demand is 0, evict.
                # If w is output, store it.
                # If w is not output, delete it.
                
                candidates = list(fast)
                # Sort by demand (ascending)
                candidates.sort(key=lambda x: demand[x])
                w = candidates[0]
                
                if w in outputs:
                    add_op("store", w)
                    slow.add(w)
                    # w is done
                    if w in unprocessed:
                        unprocessed.remove(w)
                else:
                    add_op("delete", w)
                    # w is gone. Must be recomputed later if needed.
                    # It's not in slow.
                
                fast.remove(w)
                
            # Now make u available
            if u in slow:
                add_op("load", u)
                fast.add(u)
                slow.remove(u)
            else:
                # u is not in slow. Must recompute.
                # This implies u's parents must be in fast.
                # But we are in the "no computable nodes" branch.
                # This means u's parents are NOT all in fast.
                # So we cannot compute u yet.
                # We are stuck?
                # This can happen if we deleted u, and its parents are also not in fast.
                # We need to recursively ensure u's parents are in fast.
                # To handle this simply, we can just "push" u back to the unprocessed set
                # and try to compute one of ITS parents.
                # But u is already in unprocessed? 
                # If u was deleted, it's in unprocessed.
                # We are trying to make u available to compute v.
                # If we can't make u available, we can't compute v.
                # So we should switch target to u.
                # But we already picked v.
                # Let's just add u to the "to be made available" queue.
                # For simplicity, if u is not in slow, we treat it like a new target.
                # We will loop and eventually pick u as the high priority node.
                # To avoid infinite loop, we rely on the fact that u is an ancestor of v,
                # and we will eventually load inputs.
                
                # Actually, if u is not in slow, it means we deleted it.
                # We should have kept it if demand was high.
                # If demand was low, we deleted it.
                # Now we need it again.
                # We must recompute it.
                # To recompute it, we need its parents.
                # We will just let the main loop handle it by picking u as the next target.
                # So we do nothing here?
                # If we do nothing, we are still in the "no computable" state for v.
                # The loop will pick v again?
                # We should remove v from consideration for a moment?
                # Or just pick a different parent?
                
                # Let's try to find another parent of v that is in slow or fast.
                # If none, we are stuck on v.
                # We should pick a different v.
                
                # For now, we will just break out of this specific v selection
                # and let the outer loop pick a different node.
                # To do this, we can just `continue` the while loop?
                # But we haven't made progress.
                # We need to ensure progress.
                # Progress is made when we load/compute something.
                # If we can't load u, we can't compute v.
                # We should pick a node that IS loadable.
                
                # Let's modify the selection of v.
                # Pick v such that at least one parent is in slow or fast.
                # If no such v, we are stuck (should not happen if inputs are in slow).
                
                # Let's restart the selection of v with this constraint.
                found = False
                for v_candidate in unprocessed_list:
                    # Check if any parent is in slow or fast
                    for p in parents[v_candidate]:
                        if p in slow or p in fast:
                            v = v_candidate
                            u = p
                            found = True
                            break
                    if found:
                        break
                            
                if not found:
                    # Emergency: pick any input and load it
                    for inp in inputs:
                        if inp in relevant and inp not in fast and inp in slow:
                            v = inp # dummy
                            u = inp
                            # Load u
                            if len(fast) >= fast_memory_budget:
                                # Evict
                                candidates = list(fast)
                                candidates.sort(key=lambda x: demand[x])
                                w = candidates[0]
                                if w in outputs:
                                    add_op("store", w)
                                    slow.add(w)
                                    if w in unprocessed: unprocessed.remove(w)
                                else:
                                    add_op("delete", w)
                                fast.remove(w)
                            add_op("load", u)
                            fast.add(u)
                            slow.remove(u)
                            # u is now in fast.
                            # If u was an input, it's "computed" (available).
                            # Remove from unprocessed?
                            # Inputs are not computed, they are loaded.
                            # But they are available.
                            # Let's mark them as "available" by removing from unprocessed?
                            # No, unprocessed contains nodes we need to "process" (compute or load).
                            # Once loaded, they are in fast.
                            # They are not "unprocessed" in the sense of needing computation.
                            # But they might need to be "stored" if they are outputs? Inputs are not outputs.
                            # So we can remove them from unprocessed?
                            # Yes, once an input is in fast, it's available for its children.
                            # We don't need to "compute" it again.
                            unprocessed.remove(u)
                            found = True
                            break
                            
                if not found:
                    # Deadlock
                    break

    return operations

# EVOLVE-BLOCK-END
