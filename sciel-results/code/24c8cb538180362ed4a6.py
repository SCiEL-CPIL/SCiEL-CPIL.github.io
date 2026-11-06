import math
import random
import time
from itertools import combinations

def _is_disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def _greedy_with_ordering(moduli_order, residue_strategy='first'):
    """Greedy assignment with a specific ordering and residue strategy."""
    current_pairs = []
    
    for m in moduli_order:
        forbidden = [False] * m
        
        for r_i, m_i in current_pairs:
            g = math.gcd(m, m_i)
            start = r_i % g
            for r in range(start, m, g):
                forbidden[r] = True
        
        available = [r for r in range(m) if not forbidden[r]]
        
        if available:
            if residue_strategy == 'first':
                r = available[0]
            elif residue_strategy == 'last':
                r = available[-1]
            elif residue_strategy == 'random':
                r = random.choice(available)
            else:
                r = available[0]
            current_pairs.append([r, m])
    
    return current_pairs

def _backtracking_search(moduli, deadline, max_depth=128):
    """Backtracking search to find a large set of disjoint residue classes."""
    best_pairs = []
    n = len(moduli)
    
    # Precompute gcds
    gcds = [[math.gcd(moduli[i], moduli[j]) for j in range(n)] for i in range(n)]
    
    def backtrack(index, current_pairs, forbidden_states):
        nonlocal best_pairs, deadline
        
        if time.time() > deadline:
            return
        
        if len(current_pairs) > len(best_pairs):
            best_pairs = list(current_pairs)
        
        if index >= n:
            return
        
        # Try to add moduli[index]
        m = moduli[index]
        forbidden = [False] * m
        
        for i, (r_i, m_i) in enumerate(current_pairs):
            g = gcds[index][i] if i < index else math.gcd(m, m_i)
            start = r_i % g
            for r in range(start, m, g):
                forbidden[r] = True
        
        available = [r for r in range(m) if not forbidden[r]]
        
        if available:
            # Try each available residue (limited to first few for speed)
            for r in available[:min(len(available), 5)]:
                current_pairs.append([r, m])
                backtrack(index + 1, current_pairs, forbidden_states)
                current_pairs.pop()
        
        # Try skipping moduli[index]
        backtrack(index + 1, current_pairs, forbidden_states)
    
    backtrack(0, [], None)
    return best_pairs

def _max_clique_approach(deadline):
    """
    Build a compatibility graph where vertices are (residue, modulus) pairs,
    edges connect disjoint pairs. Find a large clique using greedy + local search.
    """
    best_clique = []
    
    # Generate candidate pairs: for each modulus, consider a subset of residues
    # to keep the graph manageable
    candidates = []
    candidate_moduli = []
    
    for m in range(2, 129):
        # For small moduli, consider all residues
        # For large moduli, consider a subset
        if m <= 20:
            residues = list(range(m))
        else:
            # Sample residues
            residues = list(range(0, m, max(1, m // 10)))
            if m not in residues:
                residues.append(m - 1)
        
        for r in residues:
            candidates.append((r, m))
            candidate_moduli.append(m)
    
    n = len(candidates)
    
    # Build adjacency: two candidates are adjacent if they are disjoint
    # But we can only pick one residue per modulus
    # So we need to handle this constraint
    
    # Group candidates by modulus
    from collections import defaultdict
    by_modulus = defaultdict(list)
    for idx, (r, m) in enumerate(candidates):
        by_modulus[m].append(idx)
    
    # For each pair of moduli, precompute which residue pairs are disjoint
    moduli_list = list(by_modulus.keys())
    disjoint_pairs = {}
    
    for i, m1 in enumerate(moduli_list):
        for j, m2 in enumerate(moduli_list):
            if i >= j:
                continue
            g = math.gcd(m1, m2)
            valid = set()
            for r1 in by_modulus[m1]:
                for r2 in by_modulus[m2]:
                    if (candidates[r1][0] - candidates[r2][0]) % g != 0:
                        valid.add((r1, r2))
            disjoint_pairs[(m1, m2)] = valid
    
    # Greedy clique construction with random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        
        random.shuffle(moduli_list)
        clique = []
        chosen = {}  # modulus -> residue index
        
        for m in moduli_list:
            # Find a residue for m that is disjoint with all chosen
            valid_residues = []
            for idx in by_modulus[m]:
                r = candidates[idx][0]
                compatible = True
                for m2, idx2 in chosen.items():
                    r2 = candidates[idx2][0]
                    g = math.gcd(m, m2)
                    if (r - r2) % g == 0:
                        compatible = False
                        break
                if compatible:
                    valid_residues.append(idx)
            
            if valid_residues:
                # Pick randomly from valid
                idx = random.choice(valid_residues)
                chosen[m] = idx
                clique.append([candidates[idx][0], m])
        
        if len(clique) > len(best_clique):
            best_clique = list(clique)
    
    return best_clique

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    best_pairs = []
    
    # Strategy 1: Greedy with different orderings
    moduli = list(range(2, 129))
    
    # Try natural order
    for strategy in ['first', 'last', 'random']:
        pairs = _greedy_with_ordering(moduli, strategy)
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
    
    # Try sorted by number of divisors (fewer divisors = more flexible)
    def num_prime_factors(m):
        count = 0
        x = m
        while x % 2 == 0:
            count += 1
            x //= 2
        for p in range(3, int(math.isqrt(x)) + 1, 2):
            while x % p == 0:
                count += 1
                x //= p
        if x > 1:
            count += 1
        return count
    
    moduli_by_divisors = sorted(moduli, key=lambda m: num_prime_factors(m))
    for strategy in ['first', 'last', 'random']:
        pairs = _greedy_with_ordering(moduli_by_divisors, strategy)
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
    
    # Try random orderings
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        random.shuffle(moduli)
        pairs = _greedy_with_ordering(moduli, 'first')
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
    
    # Strategy 2: Backtracking on a subset
    if time.time() < deadline - 10:
        # Try backtracking on moduli with fewer prime factors first
        subset = moduli_by_divisors[:40]
        pairs = _backtracking_search(subset, deadline - 5)
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
    
    # Strategy 3: Max clique approach
    if time.time() < deadline - 5:
        pairs = _max_clique_approach(deadline - 2)
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
    
    return best_pairs
