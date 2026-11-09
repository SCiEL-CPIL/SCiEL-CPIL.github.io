import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s limit

    def score_set(A):
        """Calculate max(|A+A|, |A*A|) for a list of integers A."""
        n = len(A)
        sums = set()
        prods = set()
        for i in range(n):
            ai = A[i]
            for j in range(n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))

    def generate_structured_set(ap_start, ap_step, ap_len, gp_start, gp_ratio, gp_len):
        """Generate a set that is a union of an arithmetic progression and a geometric progression."""
        A = set()
        
        # Arithmetic progression
        for i in range(ap_len):
            val = ap_start + i * ap_step
            if 1 <= val <= 10000:
                A.add(val)
        
        # Geometric progression
        if gp_ratio != 1:
            curr = gp_start
            for i in range(gp_len):
                if 1 <= curr <= 10000:
                    A.add(int(round(curr)))
                curr *= gp_ratio
        
        # If we don't have 20 distinct values, fill with nearby values
        if len(A) < 20:
            vals = sorted(A)
            existing = set(A)
            # Try to add values near existing ones
            for v in vals:
                for delta in range(1, 50):
                    if len(existing) >= 20:
                        break
                    for cand in [v - delta, v + delta]:
                        if 1 <= cand <= 10000 and cand not in existing:
                            existing.add(cand)
                            A.add(cand)
                if len(existing) >= 20:
                    break
        
        # If still not enough, add random values
        if len(A) < 20:
            existing = set(A)
            while len(existing) < 20:
                cand = random.randint(1, 10000)
                if cand not in existing:
                    existing.add(cand)
                    A.add(cand)
        
        return list(A)[:20]

    def local_search(A, max_iterations=5000):
        """Perform local search to improve the set."""
        A = list(A)
        current_score = score_set(A)
        best_score = current_score
        best_A = list(A)
        no_improve = 0
        max_no_improve = 200
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            if no_improve > max_no_improve:
                break
                
            # Try a random move
            i = random.randint(0, 19)
            old_val = A[i]
            
            r = random.random()
            if r < 0.3:
                # Small additive step
                step = random.choice([-5, -3, -2, -1, 1, 2, 3, 5])
                new_val = old_val + step
            elif r < 0.5:
                # Multiplicative adjustment
                if old_val > 2:
                    div = random.choice([2, 3, 4, 5, 6])
                    new_val = old_val // div
                    if new_val < 1:
                        new_val = 1
                else:
                    new_val = old_val * random.choice([2, 3, 4, 5])
            elif r < 0.7:
                # Medium random jump
                new_val = random.randint(1, 10000)
            else:
                # Try to create collisions: pick a value that might collide with existing sums/prods
                # Sample from existing values and their combinations
                if len(A) >= 2:
                    a1 = random.choice(A)
                    a2 = random.choice(A)
                    if random.random() < 0.5:
                        new_val = a1 + a2
                    else:
                        new_val = a1 * a2
                else:
                    new_val = random.randint(1, 10000)
            
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in A and new_val != old_val:
                continue
            
            A[i] = new_val
            new_score = score_set(A)
            
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(A)
            else:
                A[i] = old_val
                no_improve += 1
        
        return best_A, best_score

    # Generate initial candidates using structured approach
    candidates = []
    
    # Various arithmetic + geometric combinations
    for ap_start in [1, 2, 3, 5, 10]:
        for ap_step in [1, 2, 3, 4, 5, 7, 10]:
            for ap_len in [5, 8, 10, 12, 15, 20]:
                for gp_start in [1, 2, 3, 5]:
                    for gp_ratio in [1.5, 2.0, 2.5, 3.0]:
                        gp_len = 20 - ap_len
                        if gp_len < 0:
                            continue
                        A = generate_structured_set(ap_start, ap_step, ap_len, gp_start, gp_ratio, gp_len)
                        if len(set(A)) == 20:
                            candidates.append(A)
    
    # Add some pure arithmetic progressions
    for start in [1, 2, 3]:
        for step in [1, 2, 3, 4, 5]:
            A = [start + i * step for i in range(20)]
            if all(1 <= x <= 10000 for x in A) and len(set(A)) == 20:
                candidates.append(A)
    
    # Add some pure geometric progressions
    for start in [1, 2, 3]:
        for ratio in [1.5, 2.0, 2.5]:
            A = []
            curr = start
            for i in range(20):
                if curr > 10000:
                    break
                A.append(int(round(curr)))
                curr *= ratio
            if len(A) >= 20 and len(set(A)) == 20:
                candidates.append(A[:20])
    
    # Add random sets for diversity
    for _ in range(20):
        candidates.append(random.sample(range(1, 10001), 20))
    
    # Add smooth number based sets
    smooths_23 = []
    for i in range(15):
        p2 = 2 ** i
        if p2 > 10000:
            break
        for j in range(15):
            p3 = 3 ** j
            val = p2 * p3
            if val > 10000:
                break
            smooths_23.append(val)
    smooths_23 = sorted(set(smooths_23))
    if len(smooths_23) >= 20:
        candidates.append(smooths_23[:20])
    
    smooths_235 = []
    for i in range(15):
        p2 = 2 ** i
        if p2 > 10000:
            break
        for j in range(12):
            p3 = 3 ** j
            val23 = p2 * p3
            if val23 > 10000:
                break
            for k in range(10):
                p5 = 5 ** k
                val = val23 * p5
                if val > 10000:
                    break
                smooths_235.append(val)
    smooths_235 = sorted(set(smooths_235))
    if len(smooths_235) >= 20:
        candidates.append(smooths_235[:20])
    
    best_score = float('inf')
    best_A = None
    
    # Process candidates with local search
    # Sort candidates by initial score to process promising ones first
    scored_candidates = []
    for A in candidates:
        A = list(set(A))
        if len(A) < 20:
            while len(A) < 20:
                A.append(random.randint(1, 10000))
            A = list(set(A))[:20]
        s = score_set(A)
        scored_candidates.append((s, A))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Process top candidates first
    processed = 0
    for init_score, A in scored_candidates:
        if time.time() > deadline:
            break
        if init_score >= best_score and processed > 10:
            # Skip if initial score is already worse than best and we've processed some
            continue
        
        # Run local search
        improved_A, improved_score = local_search(A)
        
        if improved_score < best_score:
            best_score = improved_score
            best_A = improved_A
        
        processed += 1
        if processed >= 50:
            break
    
    # If best_A is None, use the best initial candidate
    if best_A is None:
        if scored_candidates:
            best_A = scored_candidates[0][1]
        else:
            best_A = list(range(1, 21))
    
    return best_A
