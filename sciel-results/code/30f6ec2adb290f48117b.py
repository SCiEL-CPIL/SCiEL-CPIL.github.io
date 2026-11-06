import math
import time
import random
from math import gcd
from fractions import Fraction

_START = time.time()
_DEADLINE = 135.0  # safety margin under the 150s wall-clock kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _pair_ok(m1, r1, m2, r2):
    g = gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _is_valid(classes):
    for i in range(len(classes)):
        m1, r1 = classes[i]
        for j in range(i + 1, len(classes)):
            m2, r2 = classes[j]
            if not _pair_ok(m1, r1, m2, r2):
                return False
    return True

def _score(classes):
    return sum(1.0 / m for m, _ in classes)

def _find_residue(classes, m):
    """Return a residue r for modulus m that is disjoint from all existing, else None."""
    for r in range(m):
        ok = True
        for (mi, ri) in classes:
            if (r % gcd(m, mi)) == (ri % gcd(m, mi)):
                ok = False
                break
        if ok:
            return r
    return None

def _build_conflict_graph():
    """
    Build a conflict graph where nodes are (m, r) pairs.
    Two nodes conflict if they share a modulus OR if their residues are incompatible.
    We want to find a maximum weight independent set where weight = 1/m.
    
    For MWIS, we can't pick two nodes with the same m, and we can't pick two nodes
    that conflict on residues.
    
    We'll use a branch-and-bound approach with bitsets for efficiency.
    """
    # All possible (m, r) pairs
    all_pairs = []
    pair_index = {}
    for m in range(20, 201):
        for r in range(m):
            pair_index[(m, r)] = len(all_pairs)
            all_pairs.append((m, r))
    
    n = len(all_pairs)
    
    # Build conflict bitsets: conflicts[i] is a set of indices j that conflict with i
    # Two pairs conflict if:
    # 1. Same modulus m
    # 2. Different moduli m1, m2 but (r1 % gcd(m1,m2)) == (r2 % gcd(m1,m2))
    
    # For efficiency, we'll store conflicts as sorted lists or bitsets
    # Given n is about sum(m for m in 20..200) = 200*201/2 - 19*20/2 = 20100 - 190 = 19910
    # That's too large for a full bitset approach in pure Python within time limits.
    
    # Alternative: Use a smarter search. Since we can only pick at most one residue per modulus,
    # and there are 181 moduli, we can think of this as: for each modulus, choose 0 or 1 residue,
    # such that the chosen residues are pairwise compatible, maximizing sum of 1/m.
    
    # Let's use a different approach: branch on moduli in order of increasing m (higher weight first),
    # and for each modulus, try all valid residues or skip it.
    
    # Precompute for each modulus m, which residues are compatible with a given set of choices.
    # This is still complex. Let's use a simpler branch-and-bound.
    
    # Actually, let's reconsider. The key insight is that the constraint is:
    # For any two chosen (m1, r1) and (m2, r2), we need r1 % gcd(m1,m2) != r2 % gcd(m1,m2).
    
    # A practical approach: 
    # 1. Start with a good greedy solution.
    # 2. Use local search with more sophisticated moves.
    # 3. Use simulated annealing or tabu search for global exploration.
    
    return all_pairs, pair_index

def _greedy_solution():
    """Build a solution by greedily adding moduli in order of 1/m (smallest m first)."""
    classes = []
    used_moduli = set()
    
    for m in range(20, 201):
        r = _find_residue(classes, m)
        if r is not None:
            classes.append([m, r])
            used_moduli.add(m)
    
    return classes

def _local_search(classes, rng, max_time=10.0):
    """
    Local search with multiple move types:
    1. Swap: change the residue of one modulus.
    2. Add: add a new modulus with a valid residue.
    3. Remove+Add: remove one modulus, then try to add one or two others.
    4. 2-opt style: change residues of two moduli simultaneously.
    """
    t0 = time.time()
    current_score = _score(classes)
    best_classes = [list(c) for c in classes]
    best_score = current_score
    
    n = len(classes)
    
    while time.time() - t0 < max_time and _time_left() > 10:
        improved = False
        
        # Try adding missing moduli
        used = set(m for m, _ in classes)
        missing = [m for m in range(20, 201) if m not in used]
        
        for m in missing:
            if _time_left() < 10:
                break
            r = _find_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                current_score += 1.0 / m
                improved = True
        
        if improved:
            if current_score > best_score:
                best_score = current_score
                best_classes = [list(c) for c in classes]
            continue
        
        # Try residue swaps for existing moduli
        for i in range(len(classes)):
            if _time_left() < 10:
                break
            mi, ri = classes[i]
            # Try a few random alternative residues
            for _ in range(3):
                rnew = rng.randrange(mi)
                if rnew == ri:
                    continue
                classes[i][1] = rnew
                # Check validity
                ok = True
                for j in range(len(classes)):
                    if j == i:
                        continue
                    mj, rj = classes[j]
                    if not _pair_ok(mi, rnew, mj, rj):
                        ok = False
                        break
                if ok:
                    # Check if this enables adding a missing modulus
                    used = set(m for m, _ in classes)
                    for m in [m for m in range(20, 201) if m not in used]:
                        r = _find_residue(classes, m)
                        if r is not None:
                            classes.append([m, r])
                            current_score += 1.0 / m
                            break
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [list(c) for c in classes]
                    else:
                        # Revert
                        classes[i][1] = ri
                        if len(classes) > n:
                            classes.pop()
                            current_score -= 1.0 / classes[-1][0] if False else 0
                            # This is getting complicated, let's simplify
                        classes[i][1] = ri
                else:
                    classes[i][1] = ri
        
        # If no improvement, try a random restart perturbation
        if not improved and rng.random() < 0.1:
            # Remove a random modulus and re-add
            if classes:
                idx = rng.randrange(len(classes))
                removed = classes.pop(idx)
                # Try to re-add it or other moduli
                for m in range(20, 201):
                    if m != removed[0]:
                        r = _find_residue(classes, m)
                        if r is not None:
                            classes.append([m, r])
                            break
                r = _find_residue(classes, removed[0])
                if r is not None:
                    classes.append([removed[0], r])
    
    return best_classes

def _simulated_annealing(classes, rng, max_time=30.0):
    """
    Simulated annealing to escape local optima.
    """
    t0 = time.time()
    current = [list(c) for c in classes]
    current_score = _score(current)
    best = [list(c) for c in current]
    best_score = current_score
    
    temp = 0.1
    cooling = 0.999
    
    while time.time() - t0 < max_time and _time_left() > 15:
        # Generate a random neighbor
        neighbor = [list(c) for c in current]
        
        # Random move: change one residue, add a modulus, or remove a modulus
        move_type = rng.random()
        
        if move_type < 0.4 and neighbor:
            # Change a residue
            idx = rng.randrange(len(neighbor))
            mi = neighbor[idx][0]
            neighbor[idx][1] = rng.randrange(mi)
        elif move_type < 0.7:
            # Add a random missing modulus
            used = set(m for m, _ in neighbor)
            missing = [m for m in range(20, 201) if m not in used]
            if missing:
                m = rng.choice(missing)
                r = _find_residue(neighbor, m)
                if r is not None:
                    neighbor.append([m, r])
        elif move_type < 0.85 and neighbor:
            # Remove a random modulus
            idx = rng.randrange(len(neighbor))
            neighbor.pop(idx)
        else:
            # Swap two residues
            if len(neighbor) >= 2:
                i, j = rng.sample(range(len(neighbor)), 2)
                mi = neighbor[i][0]
                mj = neighbor[j][0]
                neighbor[i][1] = rng.randrange(mi)
                neighbor[j][1] = rng.randrange(mj)
        
        # Check validity
        if not _is_valid(neighbor):
            temp *= cooling
            continue
        
        new_score = _score(neighbor)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < math.exp(delta / max(temp, 1e-10)):
            current = neighbor
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = [list(c) for c in current]
        
        temp *= cooling
    
    return best

def construct():
    rng = random.Random(20240517)
    best = None
    best_score = -1.0
    
    def consider(classes):
        nonlocal best, best_score
        if classes and _is_valid(classes):
            s = _score(classes)
            if s > best_score:
                best_score = s
                best = [list(c) for c in classes]
    
    # 1) Greedy solution
    greedy = _greedy_solution()
    consider(greedy)
    
    # 2) Local search from greedy
    if _time_left() > 20:
        ls = _local_search(greedy, rng, max_time=min(15.0, _time_left() - 20))
        consider(ls)
    
    # 3) Simulated annealing from best so far
    if _time_left() > 25 and best is not None:
        sa = _simulated_annealing(best, rng, max_time=min(20.0, _time_left() - 25))
        consider(sa)
    
    # 4) Multiple random restarts with local search
    while _time_left() > 15:
        # Random order greedy
        order = list(range(20, 201))
        rng.shuffle(order)
        c = []
        for m in order:
            r = _find_residue(c, m)
            if r is not None:
                c.append([m, r])
            if _time_left() < 15:
                break
        c = _local_search(c, rng, max_time=min(5.0, _time_left() - 15))
        consider(c)
    
    # 5) Final polish: try to add any remaining moduli
    if best is not None:
        for m in range(20, 201):
            if _time_left() < 5:
                break
            if m not in [c[0] for c in best]:
                r = _find_residue(best, m)
                if r is not None:
                    best.append([m, r])
    
    if best is None:
        best = [[20, 0]]
    
    return best
