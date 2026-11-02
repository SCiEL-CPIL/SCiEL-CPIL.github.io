import time
import numpy as np
from itertools import combinations
import random

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(sorted(set(U)), dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    N = 2 * mx + 1
    
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6)
    
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def _compute_ratio_fast(S):
    """Compute |S-S|/|S+S| for a small set S using set operations."""
    S = set(S)
    sumset = set()
    diffset = set()
    for a in S:
        for b in S:
            sumset.add(a + b)
            diffset.add(a - b)
    if len(sumset) == 0:
        return 0.0
    return len(diffset) / len(sumset)

def _build_mixed_radix(atoms_seq, atoms):
    """Build the mixed-radix set from a sequence of atoms."""
    U = {0}
    max_u = 0
    m = 1
    
    for idx in atoms_seq:
        S = atoms[idx]['set']
        span = atoms[idx]['span']
        
        new_U = set()
        for u in U:
            for s in S:
                new_U.add(u + m * s)
        U = new_U
        
        if max_u == 0:
            max_u = span
        else:
            max_u = max_u + m * span
            
        m = 2 * max_u + 1
        
        if max_u > 10**18:
            break
            
    return sorted(list(U))

def _calc_analytic_score(atom_indices, atoms):
    """Calculate the analytic score for a sequence of atoms."""
    if not atom_indices:
        return 0.0
    
    log_num = 0.0
    max_u = 0
    m = 1
    
    for idx in atom_indices:
        S = atoms[idx]['set']
        log_num += atoms[idx]['log_r']
        
        span = atoms[idx]['span']
        
        if max_u == 0:
            max_u = span
        else:
            max_u = max_u + m * span
            
        m = 2 * max_u + 1
        
        if max_u > 10**18:
            return -np.inf
            
    if max_u == 0:
        return 0.0
        
    log_denom = np.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
        
    return 1.0 + log_num / log_denom

def _generate_atoms(max_span=60, max_size=8):
    """Generate a pool of small atom sets with good ratio properties."""
    atoms = []
    seen = set()
    
    # Geometric progressions
    for base in range(2, 15):
        for length in range(2, max_size + 1):
            S = [base**i for i in range(length)]
            if max(S) <= max_span:
                key = tuple(sorted(S))
                if key not in seen:
                    seen.add(key)
                    r = _compute_ratio_fast(S)
                    if r > 1.0:
                        span = max(S)
                        log_r = np.log(r)
                        log_span = np.log(max(span, 1))
                        efficiency = log_r / log_span if log_span > 0 else 0
                        atoms.append({
                            'set': S,
                            'r': r,
                            'span': span,
                            'log_r': log_r,
                            'log_span': log_span,
                            'efficiency': efficiency
                        })
    
    # Arithmetic progressions with various parameters
    for start in range(0, 5):
        for diff in range(1, 15):
            for length in range(2, max_size + 1):
                S = [start + diff * i for i in range(length)]
                if 0 not in S:
                    S.append(0)
                S = sorted(S)
                if max(S) <= max_span:
                    key = tuple(sorted(S))
                    if key not in seen:
                        seen.add(key)
                        r = _compute_ratio_fast(S)
                        if r > 1.0:
                            span = max(S)
                            log_r = np.log(r)
                            log_span = np.log(max(span, 1))
                            efficiency = log_r / log_span if log_span > 0 else 0
                            atoms.append({
                                'set': S,
                                'r': r,
                                'span': span,
                                'log_r': log_r,
                                'log_span': log_span,
                                'efficiency': efficiency
                            })
    
    # Random small sets
    for _ in range(5000):
        size = random.randint(2, min(max_size, 10))
        S = sorted(random.sample(range(max_span + 1), size))
        if 0 not in S:
            S.append(0)
        S = sorted(S)
        key = tuple(S)
        if key not in seen:
            seen.add(key)
            r = _compute_ratio_fast(S)
            if r > 1.0:
                span = max(S)
                log_r = np.log(r)
                log_span = np.log(max(span, 1))
                efficiency = log_r / log_span if log_span > 0 else 0
                atoms.append({
                    'set': S,
                    'r': r,
                    'span': span,
                    'log_r': log_r,
                    'log_span': log_span,
                    'efficiency': efficiency
                })
    
    # Mersenne-like
    for k in range(1, 10):
        S = [2**i - 1 for i in range(1, k + 1)]
        if 0 not in S:
            S.append(0)
        S = sorted(S)
        if max(S) <= max_span:
            key = tuple(S)
            if key not in seen:
                seen.add(key)
                r = _compute_ratio_fast(S)
                if r > 1.0:
                    span = max(S)
                    log_r = np.log(r)
                    log_span = np.log(max(span, 1))
                    efficiency = log_r / log_span if log_span > 0 else 0
                    atoms.append({
                        'set': S,
                        'r': r,
                        'span': span,
                        'log_r': log_r,
                        'log_span': log_span,
                        'efficiency': efficiency
                    })
    
    # Sort by efficiency
    atoms.sort(key=lambda x: x['efficiency'], reverse=True)
    
    return atoms

def _greedy_build(atoms, pool_size=50, max_atoms=20):
    """Greedily build a sequence of atoms maximizing analytic score."""
    best_seq = []
    best_score = 0.0
    pool_indices = list(range(min(pool_size, len(atoms))))
    
    for _ in range(max_atoms):
        best_add_idx = -1
        best_add_score = best_score
        
        for idx in pool_indices:
            test_seq = best_seq + [idx]
            s = _calc_analytic_score(test_seq, atoms)
            if s > best_add_score:
                best_add_score = s
                best_add_idx = idx
        
        if best_add_idx != -1:
            best_seq.append(best_add_idx)
            best_score = best_add_score
        else:
            break
    
    return best_seq, best_score

def _local_search_seq(current_seq, atoms, pool_size=50, max_iters=2000, deadline=None):
    """Local search on atom sequences."""
    pool_indices = list(range(min(pool_size, len(atoms))))
    current_score = _calc_analytic_score(current_seq, atoms)
    best_seq = current_seq[:]
    best_score = current_score
    no_improve = 0
    max_no_improve = 100
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        move_type = random.choice(['add', 'remove', 'replace', 'swap'])
        new_seq = current_seq[:]
        
        if move_type == 'add' and len(new_seq) < 25:
            idx = pool_indices[random.randint(0, len(pool_indices) - 1)]
            pos = random.randint(0, len(new_seq))
            new_seq.insert(pos, idx)
            
        elif move_type == 'remove' and len(new_seq) > 1:
            pos = random.randint(0, len(new_seq) - 1)
            new_seq.pop(pos)
            
        elif move_type == 'replace' and len(new_seq) > 0:
            pos = random.randint(0, len(new_seq) - 1)
            idx = pool_indices[random.randint(0, len(pool_indices) - 1)]
            new_seq[pos] = idx
            
        elif move_type == 'swap' and len(new_seq) > 1:
            i, j = random.sample(range(len(new_seq)), 2)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            
        if new_seq != current_seq:
            new_score = _calc_analytic_score(new_seq, atoms)
            if new_score > current_score:
                current_seq = new_seq
                current_score = new_score
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_seq = current_seq[:]
            else:
                no_improve += 1
        
        if no_improve > max_no_improve:
            break
    
    return best_seq, best_score

def _refine_gaps(U, max_iters=300, deadline=None):
    """Refine the set U using gap-based local search."""
    U = sorted(list(set(U)))
    if len(U) < 2:
        return U
    
    gaps = [U[i+1] - U[i] for i in range(len(U) - 1)]
    
    def score_from_gaps(g):
        if not g:
            return 0.0
        U_local = [0]
        for gi in g:
            U_local.append(U_local[-1] + gi)
        if U_local[-1] == 0:
            return 0.0
        return _score_set(U_local)
    
    current_gaps = gaps[:]
    current_score = score_from_gaps(current_gaps)
    best_gaps = current_gaps[:]
    best_score = current_score
    
    no_improve = 0
    max_no_improve = 80
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        move_type = random.randint(0, 4)
        new_gaps = current_gaps[:]
        
        if move_type == 0:  # Modify gap
            if len(new_gaps) > 0:
                idx = random.randint(0, len(new_gaps) - 1)
                delta = random.choice([-3, -2, -1, 1, 2, 3, 5])
                new_val = new_gaps[idx] + delta
                if new_val >= 1:
                    new_gaps[idx] = new_val
                    
        elif move_type == 1:  # Insert gap
            if len(new_gaps) < 35:
                pos = random.randint(0, len(new_gaps))
                new_gaps.insert(pos, random.randint(1, 15))
                
        elif move_type == 2:  # Remove gap
            if len(new_gaps) > 1:
                pos = random.randint(0, len(new_gaps) - 1)
                new_gaps.pop(pos)
                
        elif move_type == 3:  # Swap gaps
            if len(new_gaps) > 1:
                i, j = random.sample(range(len(new_gaps)), 2)
                new_gaps[i], new_gaps[j] = new_gaps[j], new_gaps[i]
                
        elif move_type == 4:  # Perturb multiple gaps
            if len(new_gaps) > 1:
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, len(new_gaps) - 1)
                    delta = random.choice([-2, -1, 1, 2])
                    new_val = new_gaps[idx] + delta
                    if new_val >= 1:
                        new_gaps[idx] = new_val
                
        if new_gaps != current_gaps:
            new_score = score_from_gaps(new_gaps)
            if new_score > current_score:
                current_gaps = new_gaps
                current_score = new_score
                improved = True
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_gaps = new_gaps[:]
            else:
                no_improve += 1
        
        if no_improve > max_no_improve:
            break
    
    U_best = [0]
    for g in best_gaps:
        U_best.append(U_best[-1] + g)
    
    return U_best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Step 1: Generate atoms
    atoms = _generate_atoms(max_span=60, max_size=8)
    
    if not atoms:
        return [0, 1, 2]
    
    # Step 2: Greedy initialization with multiple restarts
    best_seq = []
    best_score = 0.0
    
    for restart in range(5):
        g_seq, g_score = _greedy_build(atoms, pool_size=30 + restart * 10, max_atoms=15)
        if g_score > best_score:
            best_score = g_score
            best_seq = g_seq
    
    # Step 3: Local search on atom sequences
    deadline_local = deadline - 30.0
    while time.time() < deadline_local:
        ls_seq, ls_score = _local_search_seq(best_seq, atoms, pool_size=50, max_iters=1000, deadline=deadline_local)
        if ls_score > best_score:
            best_score = ls_score
            best_seq = ls_seq
        else:
            # Try a fresh greedy build
            g_seq, g_score = _greedy_build(atoms, pool_size=60, max_atoms=18)
            if g_score > best_score:
                best_score = g_score
                best_seq = g_seq
            else:
                break
    
    # Step 4: Construct the final set U from best_seq
    if not best_seq:
        return [0, 1]
        
    U_list = _build_mixed_radix(best_seq, atoms)
    
    # Step 5: Refinement pass using gap-based local search
    if time.time() < deadline - 5.0 and len(U_list) <= 250:
        U_list = _refine_gaps(U_list, max_iters=400, deadline=deadline - 5.0)
    
    # Ensure 0 is in the set
    if 0 not in U_list:
        U_list.insert(0, 0)
        
    return sorted(U_list)

if __name__ == "__main__":
    s = construct()
    print(f"Result Size: {len(s)}")
    if len(s) < 20:
        print(f"Result: {s}")
    else:
        print(f"Result: {s[:10]}... {s[-5:]}")
    print(f"Score: {_score_set(s)}")
