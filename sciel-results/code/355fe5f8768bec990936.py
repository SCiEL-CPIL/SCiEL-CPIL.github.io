import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0
    
    def build_sequence(scoring_func, seed=None):
        if seed is not None:
            random.seed(seed)
        seq = []
        sums_set = set()
        used = set()
        
        while True:
            candidates = []
            last = seq[-1] if seq else 0
            for x in range(last + 1, 257):
                if x in used:
                    continue
                
                # Check validity
                suf = [0]
                s = 0
                for a in reversed(seq):
                    s += a
                    suf.append(s)
                
                new_sums = [x + ss for ss in suf]
                valid = True
                for ns in new_sums:
                    if ns in sums_set:
                        valid = False
                        break
                
                if valid:
                    candidates.append(x)
            
            if not candidates:
                break
            
            # Select candidate
            if scoring_func is None:
                x = random.choice(candidates)
            else:
                scored = [(scoring_func(x, seq, candidates), x) for x in candidates]
                scored.sort(key=lambda item: item[0])
                x = scored[0][1]
            
            # Add x
            seq.append(x)
            used.add(x)
            
            # Update sums_set
            s = x
            sums_set.add(s)
            for a in reversed(seq[:-1]):
                s += a
                sums_set.add(s)
            
        return seq

    best_seq = []
    
    def score_min(x, seq, cands):
        return x
        
    def score_max(x, seq, cands):
        return -x
        
    def score_random(x, seq, cands):
        return random.random()
        
    def score_spread(x, seq, cands):
        last = seq[-1] if seq else 0
        return -(x - last)
        
    def score_central(x, seq, cands):
        med = sorted(cands)[len(cands)//2]
        return abs(x - med)

    strategies = [
        score_min,
        score_max,
        score_random,
        score_spread,
        score_central,
        None
    ]
    
    while time.time() < DEADLINE:
        for strat in strategies:
            if time.time() > DEADLINE:
                break
            seed = random.randint(0, 1000000)
            seq = build_sequence(strat, seed)
            if len(seq) > len(best_seq):
                best_seq = seq
            
    return best_seq
