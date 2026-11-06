import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    N = 256

    def check_valid(seq):
        """Return True if all non-empty consecutive block sums are distinct."""
        seen = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in seen:
                    return False
                seen.add(s)
        return True

    def compute_conflict_graph(seq):
        """
        Compute a conflict graph where vertices are elements in seq,
        and edges connect pairs that would create a sum collision if both were included.
        This is an approximation: we check if removing any pair breaks validity,
        or more practically, we use the conflict graph to guide removals.
        """
        n = len(seq)
        # For a given valid sequence, we want to identify which elements are "critical"
        # or which pairs conflict. 
        # Actually, the directive says: "edges connect pairs that would create a sum collision if both were included"
        # This is tricky because the sum collision depends on the context (the whole sequence).
        # A simpler proxy: If we have a valid sequence S, and we try to add x, it fails.
        # Why? Because some block sum in S + [x] collides with another.
        # Let's define the conflict graph dynamically during the search.
        
        # For the greedy coloring heuristic, we need a static or semi-static graph.
        # Let's build a graph where an edge (u, v) exists if u and v are "close" in value
        # or if their inclusion tends to cause collisions. 
        # A better approach: Use the current valid sequence to identify "bottlenecks".
        # If we remove element u, can we add more elements?
        
        # Let's implement a different heuristic: 
        # 1. Start with a valid sequence.
        # 2. Compute the "degree" of each element as the number of other elements
        #    that cannot be added if this element is present.
        # 3. Remove the element with the highest degree (least flexible).
        # 4. Try to add the most flexible available elements.
        
        pass

    def greedy_coloring_mis(seq, time_limit):
        """
        Greedy coloring heuristic for Maximum Independent Set.
        1. Identify a set of "candidate" elements to add.
        2. Order them by "flexibility" (lowest conflict degree).
        3. Add them one by one if valid.
        4. If stuck, remove a "high-conflict" element and retry.
        """
        current_seq = list(seq)
        current_set = set(current_seq)
        
        # Precompute all possible elements
        all_elements = list(range(1, N + 1))
        
        # Heuristic: Define "conflict degree" for an element x not in current_seq
        # as the number of elements y in current_seq such that adding x fails
        # due to a collision involving y.
        # This is expensive to compute exactly. 
        # Approximation: Use the value of x. Smaller numbers have more potential
        # collisions with larger numbers? Or vice versa?
        # Let's use a random tie-breaker and a simple heuristic:
        # Elements closer to the median of current_seq are more likely to cause collisions.
        
        def get_flexibility(x):
            """Lower is more flexible."""
            if not current_seq:
                return random.random()
            # Heuristic: Distance from the mean of current_seq
            mean_val = sum(current_seq) / len(current_seq)
            return abs(x - mean_val) + random.random() * 10  # Add noise
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Find the most flexible element to add
            candidates = [x for x in all_elements if x not in current_set]
            if not candidates:
                break
                
            # Sort by flexibility (lowest first)
            candidates.sort(key=get_flexibility)
            
            added = False
            for x in candidates:
                new_seq = sorted(current_seq + [x])
                if check_valid(new_seq):
                    current_seq = new_seq
                    current_set.add(x)
                    added = True
                    improved = True
                    break
            
            if not added:
                # We are stuck. Remove a "high-conflict" element.
                # Heuristic: Remove an element that is "central" or has many neighbors.
                # Let's try removing a random element from the middle of the sequence.
                if len(current_seq) > 1:
                    # Pick a random element to remove
                    # Or pick the one that, if removed, allows the most additions?
                    # That's expensive. Let's just pick a random one.
                    idx = random.randint(0, len(current_seq) - 1)
                    to_remove = current_seq[idx]
                    current_seq.remove(to_remove)
                    current_set.remove(to_remove)
                    improved = True  # Allow loop to continue

        return current_seq

    # Phase 1: Generate a strong initial sequence
    best_seq = []
    best_len = 0

    # Standard ascending
    seq = []
    for x in range(1, N + 1):
        new_seq = sorted(seq + [x])
        if check_valid(new_seq):
            seq = new_seq
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq[:]

    # Descending
    seq = []
    for x in range(N, 0, -1):
        new_seq = sorted(seq + [x])
        if check_valid(new_seq):
            seq = new_seq
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq[:]

    # Random shuffles
    for _ in range(20):
        if time.time() > deadline - 40:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        seq = []
        for x in order:
            new_seq = sorted(seq + [x])
            if check_valid(new_seq):
                seq = new_seq
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    # Phase 2: Greedy Coloring MIS
    if time.time() < deadline - 5:
        budget = deadline - 5 - time.time()
        num_runs = 5
        run_budget = budget / num_runs
        
        for i in range(num_runs):
            if time.time() > deadline - 5:
                break
            
            # Different initializations
            if i == 0:
                init_seq = best_seq
            else:
                # Random perturbation of best_seq
                init_seq = list(best_seq)
                k = random.randint(5, 10)
                to_remove = random.sample(init_seq, min(k, len(init_seq)))
                init_seq = [x for x in init_seq if x not in set(to_remove)]
            
            result = greedy_coloring_mis(init_seq, min(run_budget, deadline - 5 - time.time()))
            if len(result) > best_len:
                best_len = len(result)
                best_seq = result[:]

    return best_seq
