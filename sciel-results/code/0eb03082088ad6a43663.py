"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 256
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin for 60s wall-clock limit
    
    best_set = None
    best_size = float('inf')
    
    # Helper to check if adding x to a set keeps it Sidon
    def is_valid_addition(x, current_set, used_sums):
        # Check against existing elements
        for a in current_set:
            s = x + a
            if s in used_sums:
                return False
        # Check diagonal
        if 2 * x in used_sums:
            return False
        return True

    # Helper to compute the set of sums
    def get_sums(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums

    # Helper to check if a set is maximal
    def is_maximal(s):
        if not s:
            return False
        used_sums = get_sums(s)
        for x in range(1, N + 1):
            if x not in s:
                if is_valid_addition(x, s, used_sums):
                    return False
        return True

    # Greedy construction with a given order
    def greedy(order):
        selected = []
        used_sums = set()
        for x in order:
            if is_valid_addition(x, selected, used_sums):
                selected.append(x)
                # Update sums
                for a in selected[:-1]:
                    used_sums.add(x + a)
                used_sums.add(2 * x)
        return selected

    # Try to improve by removing elements if possible (not strictly necessary for greedy, but good for local search)
    # Actually, greedy already produces a maximal set? 
    # No, a greedy set is not necessarily maximal. 
    # Example: if we skip an element early because it conflicts, it might not conflict later? 
    # No, Sidon property is monotonic. If a+b is a sum, it's always a sum.
    # However, if we skip x because x+a conflicts, but later we remove a? 
    # Greedy only adds. So if x was skipped, it was because it conflicted with some existing elements.
    # Since we never remove, those conflicts remain.
    # BUT, maximality requires that NO missing element can be added.
    # If x was skipped, it means x+a was already in sums.
    # So x cannot be added to the final set either.
    # Thus, a greedy set constructed by adding elements in some order is ALWAYS maximal.
    # Let's verify this logic.
    # Suppose S is constructed by greedy.
    # Let x be an element not in S.
    # x was considered at some step.
    # If x was not added, it was because for some a in S_at_that_time, x+a in Sums_at_that_time, or 2x in Sums_at_that_time.
    # S_at_that_time is a subset of S_final.
    # Sums_at_that_time is a subset of Sums_final.
    # So the conflict persists in S_final.
    # Therefore, x cannot be added to S_final.
    # So S_final is maximal.
    
    # So the problem reduces to: find a permutation of [1..256] such that the greedy Sidon set constructed from it has minimal size.
    # This is a combinatorial optimization problem.
    # Random shuffles are a good heuristic.
    
    rng = random.Random(42)
    
    # Initial best from a few random shuffles
    # We can also try specific heuristics:
    # 1. Random
    # 2. Descending order (often yields smaller sets? Let's check. 
    #    If we pick large numbers first, the sums are large and distinct more easily? 
    #    Actually, large numbers have fewer "room" for collisions? 
    #    Let's test descending order first as a baseline.)
    
    orders_to_try = []
    
    # 1. Descending
    orders_to_try.append(list(range(N, 0, -1)))
    
    # 2. Ascending (current seed)
    orders_to_try.append(list(range(1, N + 1)))
    
    # 3. Random shuffles
    # We will generate random shuffles in the loop.
    
    def evaluate_order(order):
        s = greedy(order)
        if len(s) < best_size:
            return s, len(s)
        return None, best_size

    # Try fixed orders first
    for order in orders_to_try:
        s, size = evaluate_order(order)
        if s:
            best_set = s
            best_size = size
            
    # Random search
    # How many shuffles can we do?
    # Greedy is O(N^2) roughly. N=256. 256^2 = 65k ops.
    # Python can do ~1M-10M simple ops/sec.
    # So ~10-100 shuffles per second.
    # In 50s, we can do ~500-5000 shuffles.
    
    while time.time() < deadline:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        s = greedy(order)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s
            # Optional: Local search around this best?
            # If we find a better one, we can restart or continue.
            # Given the time, simple random restarts are robust.
            
            # If we get very small, we might want to spend more time verifying or trying to improve further.
            # But for now, just update best.
            
    return best_set
# EVOLVE-BLOCK-END
