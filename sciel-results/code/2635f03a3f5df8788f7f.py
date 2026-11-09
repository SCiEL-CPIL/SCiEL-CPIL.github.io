import random
import math

def construct():
    REQUIRED = set(range(1, 31))
    
    def is_sidon(S):
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i+1, n):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def calc_score(S):
        if not is_sidon(S):
            return float('inf')
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(S[j] - S[i])
        missing = len(REQUIRED - diffs)
        max_val = max(S)
        return 10000 * missing + max_val

    best_S = None
    best_score = float('inf')
    
    # Use a fixed seed for reproducibility if needed, but random is fine
    # random.seed(42) 
    
    # Run multiple restarts
    for restart in range(100):
        # Initialize
        S = [0]
        # Start with a small Sidon set
        # 0, 1, 3, 7, 12, 20 is a known prefix?
        # Let's just start with 0 and add greedily
        for _ in range(10):
            found = False
            # Try small values first
            for x in range(1, 50):
                if x in S:
                    continue
                new_S = sorted(S + [x])
                if is_sidon(new_S):
                    S = new_S
                    found = True
                    break
            if not found:
                # Try larger
                for x in range(50, 200):
                    if x in S:
                        continue
                    new_S = sorted(S + [x])
                    if is_sidon(new_S):
                        S = new_S
                        found = True
                        break
            if not found:
                break
        
        # Ensure size >= 9
        while len(S) < 9:
            for x in range(1, 300):
                if x in S: continue
                new_S = sorted(S + [x])
                if is_sidon(new_S):
                    S = new_S
                    break
        
        current_S = list(S)
        current_score = calc_score(current_S)
        
        # Simulated Annealing
        T = 500.0
        cooling = 0.995
        
        for step in range(2000):
            if len(current_S) <= 1:
                break
            # Pick random index (not 0)
            idx = random.randint(1, len(current_S)-1)
            old_val = current_S[idx]
            
            # Propose change
            # Small step
            if random.random() < 0.8:
                delta = random.choice([-3, -2, -1, 1, 2, 3])
                new_val = old_val + delta
            else:
                # Large step
                new_val = random.randint(1, max(old_val + 100, 100))
            
            if new_val <= 0 or new_val == old_val:
                continue
            if new_val in current_S:
                continue
            
            new_S = current_S.copy()
            new_S[idx] = new_val
            new_S.sort()
            
            new_score = calc_score(new_S)
            
            # Accept
            if new_score < current_score:
                current_S = new_S
                current_score = new_score
            else:
                dE = new_score - current_score
                if dE < 0:
                    current_S = new_S
                    current_score = new_score
                else:
                    # Avoid overflow in exp
                    if dE / T < 50: 
                        if random.random() < math.exp(-dE / T):
                            current_S = new_S
                            current_score = new_score
            
            T *= cooling
            
            if current_score < best_score:
                best_score = current_score
                best_S = list(current_S)
                
            # Check if we found a perfect solution (missing=0)
            # Score = max_val if missing=0
            # If missing=0, score is just max_val.
            # We can check if score < 10000 (since missing penalty is 10000)
            if best_score < 10000:
                break
                
        if best_S and best_score < 10000:
            # Found a valid Sidon set covering 1..30
            break

    if best_S:
        return best_S
    
    # Fallback
    S = [0]
    for x in range(1, 1000):
        if x in S: continue
        new_S = sorted(S + [x])
        if is_sidon(new_S):
            S = new_S
            if len(S) >= 12:
                break
    return S
