import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    best_set = None
    best_size = float('inf')
    
    def try_greedy(order):
        selected = []
        sums = set()
        for x in order:
            # Check if x can be added to selected
            # New sums would be {x + a for a in selected} U {2*x}
            # If any of these are in 'sums', we have a conflict.
            
            # Optimization: Check 2*x first
            if 2 * x in sums:
                continue
            
            conflict = False
            for a in selected:
                if x + a in sums:
                    conflict = True
                    break
            
            if not conflict:
                selected.append(x)
                sums.add(2 * x)
                for a in selected[:-1]:  # Add x+a for all a in previous selected
                    sums.add(x + a)
        
        return selected

    # 1. Try deterministic orders
    # Ascending
    s1 = try_greedy(range(1, N + 1))
    if len(s1) < best_size:
        best_size = len(s1)
        best_set = s1
    
    # Descending
    if time.time() < deadline:
        s2 = try_greedy(range(N, 0, -1))
        if len(s2) < best_size:
            best_size = len(s2)
            best_set = s2

    # 2. Random restarts
    # We can do many of these. Each takes ~O(N * size) which is small.
    # Estimate: 256 * 10 = 2560 ops. 10,000 restarts = 25.6M ops.
    # Python might do ~10M ops/sec. So 10,000 restarts ~ 2.5s.
    # We have 50s. Let's do as many as possible.
    
    while time.time() < deadline:
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = try_greedy(order)
        if len(s) < best_size:
            best_size = len(s)
            best_set = s
            # If we find size 1, it's impossible for Sidon? 
            # Min size is likely > 1.
            # If best_size reaches a theoretical lower bound, we could stop, 
            # but we don't know it exactly. 
            # However, if we find 7, it's a big jump.
        
        # Optional: Local search?
        # Try to remove elements from best_set and see if it stays maximal?
        # If it stays maximal, we have a smaller set!
        # But checking maximality is expensive (try all missing elements).
        # Given N=256, checking maximality is O(N * |S| * cost_check).
        # Cost check is O(|S|). So O(N * |S|^2).
        # 256 * 100 = 25600. Very fast.
        # So we can verify maximality of any candidate.
        # Wait, the greedy construction *guarantees* maximality?
        # Yes, as argued before: if we skipped x, it conflicted with S_t.
        # S_t subset S_final. So x conflicts with S_final.
        # So greedy output is always maximal.
        # So we don't need to verify maximality, just size.
        
    if best_set is None:
        # Fallback to seed behavior
        return try_greedy(range(1, N + 1))
        
    return best_set
# EVOLVE-BLOCK-END
