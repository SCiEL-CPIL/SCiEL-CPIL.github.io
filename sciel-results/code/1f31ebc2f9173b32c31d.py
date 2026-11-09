import time
import random
import sys
from itertools import combinations

N_POINTS = 16
DEADLINE = 140.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Precompute all possible triples
    all_triples = list(combinations(range(N_POINTS), 3))
    
    # Helper: check if a triple is compatible with a set of used pairs
    def is_compatible(triple, used_pairs):
        a, b, c = triple
        return (a, b) not in used_pairs and (a, c) not in used_pairs and (b, c) not in used_pairs
    
    # Helper: compute used pairs from a system
    def get_used_pairs(system):
        pairs = set()
        for t in system:
            a, b, c = t
            pairs.add((a, b))
            pairs.add((a, c))
            pairs.add((b, c))
        return pairs
    
    # Helper: compute independence number exactly using bitmask DP
    # Since N_POINTS=16, we can use a bitmask of size 2^16 = 65536
    # alpha(system) = max { |S| : no triple in system is subset of S }
    # Equivalent: S is independent if for every triple t in system, not (t subset of S)
    
    def compute_alpha(system):
        if not system:
            return N_POINTS
        
        # Create a bitmask for each triple
        triple_masks = []
        for t in system:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        # We want to find the largest S (bitmask) such that no triple_mask is a subset of S
        # That is, for all m in triple_masks: (m & S) != m
        
        # We can iterate over all subsets? 2^16 = 65536, which is feasible.
        # But we can do better: start from full set and remove? No, that's not exact.
        # Let's just iterate over all subsets in descending order of size.
        # Actually, we can use a simple loop over all masks from 0 to 2^16-1,
        # but that's 65536 iterations, each checking up to ~40 triples. 65536*40 = 2.6M ops, fast.
        
        best = 0
        for S in range(1 << N_POINTS):
            # Check if S is independent
            independent = True
            for m in triple_masks:
                if (m & S) == m:
                    independent = False
                    break
            if independent:
                cnt = bin(S).count('1')
                if cnt > best:
                    best = cnt
                    # We can't break early because we want the max
        return best
    
    # Initial system: use greedy to get a large linear triple system
    best_initial = []
    best_size = 0
    for _ in range(100):
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        selected = []
        used_pairs = set()
        for t in shuffled:
            if is_compatible(t, used_pairs):
                selected.append(t)
                a, b, c = t
                used_pairs.add((a, b))
                used_pairs.add((a, c))
                used_pairs.add((b, c))
        if len(selected) > best_size:
            best_size = len(selected)
            best_initial = selected[:]
    
    current_system = best_initial[:]
    if not current_system:
        current_system = [(i*3, i*3+1, i*3+2) for i in range(5)]
    
    current_alpha = compute_alpha(current_system)
    best_system = current_system[:]
    best_alpha = current_alpha
    
    # Local search: try to replace one triple with another to reduce alpha
    # We will use a simple hill-climbing with random restarts if stuck
    
    # Precompute for each triple, which other triples are incompatible (share a pair)
    # This is static for a given system, but changes as we swap. So we recompute each time.
    
    # To speed up, we can precompute for each point, the triples containing it? 
    # But the system changes. Let's just do direct checks.
    
    # Strategy: 
    # 1. Pick a random triple to remove.
    # 2. Find all triples that are compatible with the remaining system.
    # 3. Among those, pick a few candidates (e.g., random or based on some heuristic) and evaluate alpha.
    # 4. If any candidate gives lower alpha, accept it.
    # 5. If stuck, do a random restart or accept a worse solution temporarily (SA-like).
    
    # Let's use a simple SA-like approach but with exact alpha evaluation.
    # Since alpha evaluation is ~2.6M ops, we can do maybe ~500-1000 evaluations in 120s? 
    # 2.6M ops in Python might take ~0.1-0.5s. So ~200-1000 evals in 120s. That's enough.
    
    T = 5.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Precompute a list of all triples for random selection
    # We already have all_triples
    
    while time.time() - start_time < DEADLINE:
        if T < min_T:
            break
        
        if not current_system:
            break
        
        # Pick a random index to swap
        idx = random.randint(0, len(current_system) - 1)
        old_triple = current_system[idx]
        
        # Remove it
        temp_system = current_system[:]
        temp_system.pop(idx)
        used_pairs = get_used_pairs(temp_system)
        
        # Find all compatible triples
        # To speed up, we can pre-filter? But let's just check all.
        # There are C(16,3)=560 triples. Checking each against used_pairs is fast.
        compatible = []
        for t in all_triples:
            if is_compatible(t, used_pairs):
                compatible.append(t)
        
        if not compatible:
            # No move possible, break or restart
            break
        
        # Pick a random candidate from compatible
        # Or pick the one that minimizes alpha? That would be expensive.
        # Let's pick a random one, or a few.
        
        # For efficiency, let's try up to 5 random candidates
        candidates = random.sample(compatible, min(5, len(compatible)))
        
        best_candidate = None
        best_candidate_alpha = current_alpha + 100  # worse than current
        
        for cand in candidates:
            # Construct new system
            new_system = temp_system[:]
            new_system.append(cand)
            # Compute alpha
            new_alpha = compute_alpha(new_system)
            if new_alpha < best_candidate_alpha:
                best_candidate_alpha = new_alpha
                best_candidate = cand
        
        if best_candidate is not None:
            # Evaluate if we accept
            delta = best_candidate_alpha - current_alpha
            if delta < 0:
                # Better, accept
                current_system[idx] = best_candidate
                current_alpha = best_candidate_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_system = current_system[:]
            else:
                # Worse or equal, accept with probability exp(-delta/T)
                if delta == 0:
                    current_system[idx] = best_candidate
                    current_alpha = best_candidate_alpha
                else:
                    if delta < 20 * T:
                        if random.random() < pow(2.718281828, -delta / T):
                            current_system[idx] = best_candidate
                            current_alpha = best_candidate_alpha
        
        T *= cooling_rate
        
    return [list(t) for t in best_system]

# Fallback in case of any error
if __name__ == "__main__":
    sys = construct()
    print(f"System size: {len(sys)}")
    # Verify linearity
    for i in range(len(sys)):
        for j in range(i+1, len(sys)):
            common = set(sys[i]) & set(sys[j])
            if len(common) > 1:
                print("Invalid system!")
                break
