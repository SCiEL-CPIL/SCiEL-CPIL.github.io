import time
import random

def construct():
    start_time = time.time()
    timeout = 100.0  # Safety margin (evaluator kills at 150s)
    
    def evaluate_fast(S):
        """Fast evaluation using bitset representation."""
        # Create a bitset where bit i is set if i is in S
        bits = 0
        for x in S:
            bits |= (1 << x)
        
        valid_diffs = set()
        # S is assumed to be a list of 32 distinct integers
        # We check all pairs (a, b) where a < b in the list order? 
        # No, a, a+d, a+2d. 
        # Let's iterate over all pairs of distinct elements in the set.
        # To avoid duplicates and ensure a < a+d, we just check all pairs (x, y) from the set.
        # If x < y, let d = y - x. Check if x + 2d is in the set.
        # Since the set is small (32), N^2 = 1024 checks max.
        # Bitset check: (bits >> val) & 1
        
        # Sort S to make it deterministic? Not strictly needed for set logic, but good for consistency.
        # Actually, we don't need to sort for the logic, just iterate the list.
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(n):
                if i == j:
                    continue
                b = S[j]
                if a > b:
                    continue # We want a < b to define d > 0
                d = b - a
                c = a + 2 * d
                if c > 512:
                    continue
                if (bits >> c) & 1:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    def random_init():
        return random.sample(range(1, 513), 32)
    
    def convex_init():
        pts = [1]
        for i in range(1, 32):
            gap = 1 + i
            pts.append(pts[-1] + gap)
        return pts
    
    def structured_block_init():
        """Create set with blocks of consecutive numbers separated by gaps."""
        # Try different block structures
        candidates = []
        
        # 4 blocks of 8
        for offset in range(0, 50, 5):
            pts = []
            for block in range(4):
                start = 1 + block * 130 + offset
                for k in range(8):
                    if start + k <= 512:
                        pts.append(start + k)
            if len(pts) == 32 and len(set(pts)) == 32:
                candidates.append(pts)
        
        # 8 blocks of 4
        for offset in range(0, 50, 5):
            pts = []
            for block in range(8):
                start = 1 + block * 65 + offset
                for k in range(4):
                    if start + k <= 512:
                        pts.append(start + k)
            if len(pts) == 32 and len(set(pts)) == 32:
                candidates.append(pts)
                
        if candidates:
            return random.choice(candidates)
        return random_init()
    
    def geometric_init():
        """Geometric progression based."""
        pts = []
        base = 2
        for i in range(32):
            val = 1 + int(base ** i)
            if val > 512:
                val = 512 - (31 - i)
            if val not in pts and val >= 1 and val <= 512:
                pts.append(val)
            else:
                # Fill remaining
                while len(pts) < 32:
                    v = random.randint(1, 512)
                    if v not in pts:
                        pts.append(v)
        return pts[:32]
    
    # Generate diverse initial candidates
    candidates = [
        list(range(1, 33)),       # Consecutive
        convex_init(),            # Convex
        random_init(),
        random_init(),
        random_init(),
        structured_block_init(),
        structured_block_init(),
        geometric_init(),
    ]
    
    # Evaluate initial candidates
    best_set = None
    best_score = -1
    for c in candidates:
        if len(set(c)) != 32:
            continue
        s = evaluate_fast(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    current_set = best_set[:]
    current_score = best_score
    
    # Simulated Annealing with multiple move types
    T = 3.0
    T_min = 0.1
    cooling = 0.9995
    
    # Pre-calculate exp table? No, T changes. 
    # But we can optimize the inner loop.
    
    def generate_neighbor(S, move_type):
        """Generate a neighbor using different move types."""
        new_set = S[:]
        if move_type == 0:
            # Swap two elements
            i, j = random.sample(range(32), 2)
            new_set[i], new_set[j] = new_set[j], new_set[i]
        elif move_type == 1:
            # Replace one element
            i = random.randrange(32)
            new_val = random.randint(1, 512)
            if new_val in new_set:
                return None
            new_set[i] = new_val
        elif move_type == 2:
            # Double replace
            i, j = random.sample(range(32), 2)
            new_val1 = random.randint(1, 512)
            new_val2 = random.randint(1, 512)
            if new_val1 in new_set or new_val2 in new_set:
                return None
            new_set[i] = new_val1
            new_set[j] = new_val2
        return new_set
    
    while time.time() < start_time + timeout:
        # Choose move type: swap (40%), replace (40%), double-replace (20%)
        r = random.random()
        if r < 0.4:
            move_type = 0
        elif r < 0.8:
            move_type = 1
        else:
            move_type = 2
            
        new_set = generate_neighbor(current_set, move_type)
        if new_set is None or len(set(new_set)) != 32:
            continue
            
        new_score = evaluate_fast(new_set)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(delta/T) if worse
        # Use math.exp for speed if numpy is not strictly needed for this small scalar
        # But numpy is available. Let's stick to standard lib for safety/speed in scalar
        if delta > 0:
            accept = True
        else:
            # exp(delta/T) where delta <= 0
            # If T is small, exp is small.
            import math
            accept = random.random() < math.exp(delta / T)
            
        if accept:
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # If stuck at minimum temperature, restart with higher T occasionally
        if T <= T_min and random.random() < 0.01:
            T = 1.0
    
    return best_set
