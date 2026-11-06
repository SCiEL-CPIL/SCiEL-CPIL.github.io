"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _score(S):
    """
    Count distinct d > 0 such that there exists a with a, a+d, a+2d in S.
    Optimized for speed with N=32.
    """
    S_set = set(S)
    diffs = set()
    # Local variables for faster access
    s_set = S_set
    # Iterate over all pairs to find potential differences
    # We can restrict d to be positive
    for a in S:
        for b in S:
            d = b - a
            if d > 0:
                # Check if a+2d is in S
                # a+2d = a + (b-a) + (b-a) = b + (b-a)
                # This is just a + 2*d
                if (a + 2 * d) in s_set:
                    diffs.add(d)
    return len(diffs)

def _simulated_annealing(initial_S, time_limit):
    """
    Run Simulated Annealing to maximize the score.
    """
    S = list(initial_S)
    current_score = _score(S)
    best_S = list(S)
    best_score = current_score
    
    # SA Parameters
    # Heuristic: Start T high enough to accept most bad moves, cool down.
    # Score range is roughly 0 to ~100? Let's assume max score < 200.
    # If delta score is ~5, we want P(accept) ~ 0.5 at start -> T ~ 5/ln(2) ~ 7.
    # Let's start higher to explore more, e.g., T=20.
    T_start = 20.0
    T_end = 0.01
    # We don't know total steps, so use time-based cooling or fixed steps with time check.
    # Let's estimate steps. _score is ~1000 ops. 10000 steps/s -> 10000 * 1000 = 10M ops.
    # 150s budget. Let's aim for ~200,000 iterations if possible.
    
    start_time = time.time()
    elapsed = 0.0
    
    # Pre-calculate cooling factor if we knew total steps, but we don't.
    # Use exponential decay based on elapsed time?
    # T(t) = T_start * exp(-k * t)
    # We want T(T_limit) = T_end
    # T_end = T_start * exp(-k * T_limit)
    # k = ln(T_start / T_end) / T_limit
    
    # We don't know exact T_limit in advance for this specific call, but we pass time_limit.
    k = math.log(T_start / T_end) / time_limit
    
    it = 0
    while elapsed < time_limit:
        # Cooling
        T = T_start * math.exp(-k * elapsed)
        
        # Generate a move: Swap one element in S with one random element not in S
        idx = random.randint(0, 31)
        old_val = S[idx]
        
        # Pick a new value not in S
        # Efficient way: pick random from 1..512, if in S, retry? 
        # Or maintain a list of available? 
        # For N=32, probability of collision is 32/512 ~ 6%. Retry is cheap.
        new_val = random.randint(1, 512)
        while new_val == old_val or new_val in S:
            new_val = random.randint(1, 512)
            
        # Apply move
        S[idx] = new_val
        new_score = _score(S)
        
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_S = list(S)
        else:
            # Revert
            S[idx] = old_val
            
        it += 1
        # Check time every 1000 iterations to avoid overhead of time.time() every loop
        if it % 1000 == 0:
            elapsed = time.time() - start_time
            
    return best_S, best_score

def _hill_climb_polish(S, time_limit=5.0):
    """
    Strict hill climbing to polish the result.
    """
    S = list(S)
    current_score = _score(S)
    best_S = list(S)
    best_score = current_score
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try all possible swaps? That's 32 * 480 = 15360 swaps per full pass.
        # Too slow for a full pass every iteration.
        # Instead, do random local search steps but only accept improvements.
        idx = random.randint(0, 31)
        old_val = S[idx]
        new_val = random.randint(1, 512)
        
        if new_val == old_val or new_val in S:
            continue
            
        S[idx] = new_val
        new_score = _score(S)
        
        if new_score > current_score:
            current_score = new_score
            best_S = list(S)
            best_score = new_score
            improved = True
        else:
            S[idx] = old_val
            
    return best_S, best_score

def construct():
    """
    Construct a 32-element subset of 1..512 maximizing the number of distinct
    3-term arithmetic progression differences.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_S = None
    best_score = 0
    
    # Candidate Initializations
    # 1. Consecutive
    S1 = list(range(1, 33))
    # 2. Stratified Random (good spread)
    # Divide 1..512 into 32 buckets of 16, pick one from each
    S2 = []
    for i in range(32):
        start = i * 16 + 1
        end = (i + 1) * 16
        S2.append(random.randint(start, end))
    S2 = sorted(set(S2))
    # Ensure 32 distinct (if collision, fix)
    while len(S2) < 32:
        S2.append(random.randint(1, 512))
        S2 = sorted(set(S2))
    
    # 3. Random
    S3 = random.sample(range(1, 513), 32)
    
    candidates = [S1, S2, S3]
    
    # Evaluate candidates
    for S in candidates:
        s = _score(S)
        if s > best_score:
            best_score = s
            best_S = list(S)
            
    # Run SA on the best candidate and a few others if time permits
    # We have ~145s. 
    # Let's run 3 SA instances of ~40s each.
    
    sa_runs = 3
    sa_time_each = (deadline - (time.time() - start_time)) / sa_runs - 5.0 # 5s buffer for polish
    
    for i in range(sa_runs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
            
        # Choose a starting point
        # First run: best found so far
        # Second run: S2 (stratified)
        # Third run: S3 (random)
        if i == 0:
            init_S = best_S
        elif i == 1:
            init_S = S2
        else:
            init_S = S3
            
        # Run SA
        S_res, score_res = _simulated_annealing(init_S, sa_time_each)
        
        if score_res > best_score:
            best_score = score_res
            best_S = list(S_res)
            
    # Final Polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        polish_time = min(10.0, deadline - elapsed - 1.0)
        if polish_time > 0:
            best_S, best_score = _hill_climb_polish(best_S, polish_time)
            
    # Final Validation
    result = sorted(set(best_S))
    if len(result) != 32:
        # Fallback
        result = list(range(1, 33))
        
    return result

# EVOLVE-BLOCK-END
