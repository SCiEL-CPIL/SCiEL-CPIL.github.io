import numpy as np
import time

def priority(adjacency):
    """
    Strategy: Simulated Annealing on the decoder's decision sequence.
    
    Instead of assigning static scores and letting a fixed greedy decoder pick,
    we directly optimize the set of vertices that will be selected.
    
    We fix a vertex ordering (by ascending degree, then index) and simulate the
    decoder's traversal. At each step, the decoder either "picks" the vertex 
    (if it's not blocked) or "skips" it. We represent the solution as a bitmask 
    of picks. 
    
    Moves: Flip a bit in the decision mask (subject to validity constraints).
    Objective: Size of the resulting independent set.
    
    This is a different design family: we are not searching for scores, but for 
    the optimal sequence of greedy decisions, using SA to escape local optima 
    in the decision space.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin

    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=np.float64)

    # Precompute neighbors as sets for fast intersection
    # For speed in the inner loop, we use numpy arrays or lists
    # Let's use lists of lists for neighbors
    neighbors = [np.where(adj[i])[0].tolist() for i in range(n)]
    deg = np.array([len(neighbors[i]) for i in range(n)], dtype=np.int32)

    # Fixed ordering: Ascending degree, then index.
    # This mimics the "pick low degree first" heuristic which is often good for MIS.
    order = np.argsort(deg, kind='stable')
    
    # We will represent the solution as a boolean array `decisions` of length n.
    # decisions[v] = True means "when the decoder reaches v, pick it".
    # The decoder processes vertices in `order`.
    # If a vertex is picked, its neighbors are blocked.
    
    def evaluate(decisions):
        """
        Given a decision mask, compute the independent set size and the set of picked vertices.
        The decoder iterates through `order`.
        If decisions[v] is True AND v is not blocked, pick v.
        """
        picked = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        
        for v in order:
            if decisions[v] and not blocked[v]:
                picked[v] = True
                count += 1
                for u in neighbors[v]:
                    if not picked[u]:
                        blocked[u] = True
        
        return count, picked, blocked

    # Initialize with a greedy heuristic: Pick if not blocked (equivalent to decisions=True for all)
    # Actually, let's start with a smart random or greedy start.
    # Greedy start: decisions[v] = True for all v.
    # This results in the standard greedy MIS on the `order`.
    current_decisions = np.ones(n, dtype=bool)
    current_size, current_picked, _ = evaluate(current_decisions)
    
    best_decisions = current_decisions.copy()
    best_size = current_size
    best_picked = current_picked.copy()
    
    # Simulated Annealing Parameters
    # We need to define a move.
    # Move: Flip decisions[v] for a random v.
    # But wait: flipping a bit might not change the outcome if v is blocked anyway.
    # Also, if we flip a bit from False to True, and v is not blocked, we pick it.
    # If we flip from True to False, we skip it.
    
    # A more effective move structure for this specific greedy-decoder constraint:
    # Since the decoder is fixed, the "state" is the decision mask.
    # However, simple bit flips are slow to explore.
    
    # Alternative Move: "Re-insertion" or "Swap" in the picked set?
    # No, the decoder is fixed. We must stick to decision flips.
    
    # To make SA effective, we need a good temperature schedule and move selection.
    # Let's try: Pick a random vertex v. Flip decisions[v].
    # Calculate delta in size.
    
    # Optimization: Fast delta calculation.
    # When we flip decisions[v]:
    # 1. If we set to True (was False):
    #    - If v is blocked in current state, no change.
    #    - If v is not blocked, v is picked.
    #      - This blocks all neighbors of v that were previously picked? No.
    #      - The decoder is sequential. Changing a decision for v affects:
    #        a) v itself (picked or not).
    #        b) Any subsequent vertex u in `order` that is a neighbor of v.
    #           If u was picked before, it might now be blocked.
    #           If u was not picked (blocked or decision=False), it might remain so.
    #      - This "forward propagation" makes delta calculation complex and potentially O(n) per move.
    
    # Given n=200, O(n) per move is acceptable. We can afford ~10^5 - 10^6 moves.
    
    def get_delta(v, new_val, decisions, picked, blocked):
        """
        Estimate the change in IS size if we flip decisions[v] to new_val.
        We re-simulate the decoder from v onwards to get the exact delta.
        Since n is small (200), this is cheap.
        """
        # We need to know the state of `blocked` and `picked` up to v.
        # But `blocked` and `picked` depend on previous decisions.
        # If we change decisions[v], it only affects v and vertices AFTER v in `order`.
        
        # Let's find the index of v in `order`.
        # We need a map: vertex -> position in order
        pos_in_order = np.zeros(n, dtype=np.int32)
        for i, v_idx in enumerate(order):
            pos_in_order[v_idx] = i
        
        pos_v = pos_in_order[v]
        
        # Re-simulate from scratch? Or from start?
        # Re-simulating from scratch is O(n * deg). n=200, deg~5. 1000 ops.
        # 100,000 moves * 1000 ops = 10^8 ops. Might be tight in Python.
        # Let's try to be smarter.
        
        # Actually, let's just re-evaluate the whole thing if we accept the move.
        # For delta, we can do a partial re-evaluation.
        
        # State before v:
        # We need the `blocked` status of all vertices BEFORE v is processed.
        # This is invariant to the change at v (since v is processed later).
        # So we can pre-compute the prefix state?
        # But the prefix state depends on decisions[0..pos_v-1].
        # If we only flip decisions[v], the prefix is unchanged.
        
        # So:
        # 1. Compute prefix picked/blocked up to pos_v - 1.
        # 2. Simulate from pos_v to end with the NEW decision for v.
        # 3. Compare count with current_size.
        
        # To avoid O(n) prefix computation every time, we can store the full picked/blocked arrays.
        # But if we flip v, the suffix changes. The prefix remains the same.
        
        # Let's implement a fast partial evaluation.
        
        # Prefix state (invariant):
        # We need `blocked_prefix` and `picked_prefix` for vertices processed before v.
        # Actually, we just need the `blocked` status of vertices that are neighbors of v or come after v.
        
        # Simpler approach: Just run the full evaluation for the candidate.
        # It's O(n^2) worst case, but with n=200, it's fast in NumPy/Python if optimized.
        
        # Let's optimize `evaluate` to be as fast as possible.
        
        # Actually, let's just use the full evaluation for simplicity first, and see if time allows.
        # If not, we optimize.
        
        # For now, return the candidate decisions to be evaluated fully.
        pass

    # Let's implement a faster delta by re-simulating only the suffix.
    # We need the state of `blocked` for all vertices at the moment v is considered.
    # Let's pre-compute the "prefix blocked" array for the current decisions?
    # No, because if we change a decision at v, it doesn't change the prefix.
    # So, for a given current state, the prefix up to v is fixed.
    
    # Let's create a helper to get the state at position k in the order.
    # This is expensive to do from scratch.
    
    # Alternative: Just use full evaluation.
    # n=200.
    # Loop: 200 iterations.
    # Inner loop: average degree 5.
    # Total ~1000 operations per evaluation.
    # Python overhead ~1us per op? 1ms per eval.
    # 10,000 evals = 10s.
    # We have 145s. So we can do ~100,000 evals.
    # This is enough for SA.
    
    # SA Loop
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    # Pre-compute pos_in_order for fast access
    pos_in_order = np.zeros(n, dtype=np.int32)
    for i, v_idx in enumerate(order):
        pos_in_order[v_idx] = i
        
    # We'll use a simple random walk on the decision bits.
    # To make it more effective, we can bias towards flipping bits that are "active" (i.e., v is not blocked).
    
    while time.time() - start_time < deadline:
        if T < T_min:
            T = T_min
            
        # Pick a random vertex to flip
        v = np.random.randint(n)
        
        # New decision
        new_decisions = current_decisions.copy()
        new_decisions[v] = not current_decisions[v]
        
        # Evaluate new size
        new_size, _, _ = evaluate(new_decisions)
        
        delta = new_size - current_size
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta >= 0 or np.random.random() < np.exp(-delta / T):
            current_decisions = new_decisions
            current_size = new_size
            
            if current_size > best_size:
                best_size = current_size
                best_decisions = current_decisions.copy()
                _, best_picked, _ = evaluate(best_decisions)
                
        T *= cooling
        
    # Convert best_decisions to scores.
    # The decoder picks in descending score order.
    # We want the decoder to pick exactly the vertices in `best_picked`.
    # How to construct scores?
    # If we assign high scores to picked vertices and low scores to others,
    # the decoder will pick the high scores first.
    # However, the decoder skips neighbors.
    # If we give score 1 to picked vertices and 0 to others:
    # The decoder will pick all score-1 vertices that are not blocked by other score-1 vertices.
    # Since `best_picked` is an independent set, no two picked vertices are neighbors.
    # So if we assign score 1 to all `best_picked` vertices, the decoder will pick ALL of them,
    # provided that no non-picked vertex with score 1 blocks them.
    # But we only assign score 1 to picked vertices.
    # What about vertices not in `best_picked`? They get score 0.
    # The decoder picks in descending score.
    # All picked vertices have score 1. All others have score 0.
    # The decoder will process all score-1 vertices first.
    # Since they form an independent set, none block each other.
    # So the decoder will pick all of them.
    # Then it processes score-0 vertices.
    # Any score-0 vertex that is not blocked by a picked vertex could be picked.
    # This might add extra vertices to the independent set, increasing the size!
    # Is that bad? No, higher is better.
    # But wait, if the decoder picks extra vertices, the size is even larger than `best_size`.
    # So this is a valid lower bound on the achievable size.
    # Actually, we want the scores to guide the decoder to produce *at least* the `best_picked` set.
    # Assigning 1 to picked and 0 to others guarantees `best_picked` is included.
    # It might include more, which is fine.
    
    scores = np.zeros(n, dtype=np.float64)
    scores[best_picked] = 1.0
    
    # To break ties or refine, we can add a small epsilon based on degree or index.
    # But 1.0 and 0.0 are distinct.
    
    return scores

