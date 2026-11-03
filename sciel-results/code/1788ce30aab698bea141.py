import time
import random
import math
from itertools import combinations

def _score_set_fft(U):
    """Fast scoring using FFT for sumset and difference set sizes."""
    if not U or 0 not in U:
        return -1
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # For small sets, direct computation is faster
    if len(U) <= 30:
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        s_size = len(sums)
        d_size = len(diffs)
    else:
        # Use numpy FFT for larger sets
        import numpy as np
        n = len(U)
        max_val = max(U)
        size = 2 * max_val + 1
        
        # Create indicator vector
        vec = np.zeros(size, dtype=np.float64)
        for u in U:
            vec[u] = 1.0
        
        # FFT to compute sumset (convolution)
        fft_size = 1
        while fft_size < 2 * size:
            fft_size *= 2
        
        f = np.fft.fft(vec, fft_size)
        conv = np.fft.ifft(f * f).real
        s_size = int(np.sum(conv > 0.5))
        
        # Difference set: correlate vec with itself
        # Or use: |U-U| = number of distinct (a-b) for a,b in U
        # This is the autocorrelation
        corr = np.fft.ifft(f * np.conj(f)).real
        d_size = int(np.sum(corr > 0.5))
    
    if s_size == 0 or d_size == 0:
        return -1
        
    ratio = d_size / s_size
    if ratio <= 1:
        return 1.0 if ratio == 1 else 0.0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def score_set(U):
    """Score function - uses FFT for larger sets."""
    if not U or 0 not in U:
        return -1
    U = sorted(set(U))
    if len(U) <= 30:
        # Direct computation for small sets
        max_u = max(U)
        if max_u == 0:
            return 1.0
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        s_size = len(sums)
        d_size = len(diffs)
        if s_size == 0 or d_size == 0:
            return -1
        ratio = d_size / s_size
        if ratio <= 1:
            return 1.0 if ratio == 1 else 0.0
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
        return 1.0 + math.log(ratio) / denom
    else:
        return _score_set_fft(U)

def _golomb_ruler(n):
    """Generate a Golomb ruler with n marks using greedy + local search."""
    if n <= 1:
        return [0]
    
    # Start with greedy construction
    U = [0]
    for i in range(1, n):
        # Try to place next mark as far as possible while maintaining Golomb property
        best_pos = 1
        best_valid = False
        for pos in range(1, 1000):
            candidate = [0] + U[1:] + [pos]
            # Check if all differences are unique
            diffs = []
            valid = True
            for i2 in range(len(candidate)):
                for j2 in range(i2 + 1, len(candidate)):
                    d = candidate[j2] - candidate[i2]
                    if d in diffs:
                        valid = False
                        break
                    diffs.append(d)
                if not valid:
                    break
            if valid:
                best_pos = pos
                best_valid = True
                if pos > 50:  # Don't search too far
                    break
        
        if best_valid:
            U.append(best_pos)
        else:
            # Fallback: add at end
            U.append(U[-1] + 1)
    
    # Local search to improve
    random.seed(42)
    for _ in range(1000):
        U_copy = U[:]
        i = random.randint(1, n - 1)
        old_val = U_copy[i]
        new_val = random.randint(1, 500)
        if new_val in U_copy:
            continue
        U_copy[i] = new_val
        U_copy.sort()
        
        # Check Golomb property
        diffs = set()
        valid = True
        for i2 in range(len(U_copy)):
            for j2 in range(i2 + 1, len(U_copy)):
                d = U_copy[j2] - U_copy[i2]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            if not valid:
                break
        
        if valid:
            U = U_copy
    
    return sorted(U)

def _sidon_ruler(n):
    """Generate a Sidon set (B2 sequence) with n elements."""
    if n <= 1:
        return [0]
    
    U = [0]
    for i in range(1, n):
        # Try to find position that maintains Sidon property
        best_pos = 1
        found = False
        for pos in range(1, 500):
            candidate = sorted(U + [pos])
            # Check all pairwise sums are unique (counting multiplicity correctly)
            sums = []
            valid = True
            for i2 in range(len(candidate)):
                for j2 in range(i2, len(candidate)):
                    s = candidate[i2] + candidate[j2]
                    if s in sums:
                        valid = False
                        break
                    sums.append(s)
                if not valid:
                    break
            if valid:
                best_pos = pos
                found = True
                break
        
        if found:
            U.append(best_pos)
        else:
            U.append(U[-1] + 1)
    
    # Local search
    random.seed(42)
    for _ in range(500):
        U_copy = U[:]
        i = random.randint(1, n - 1)
        new_val = random.randint(1, 500)
        if new_val in U_copy:
            continue
        U_copy[i] = new_val
        U_copy.sort()
        
        # Check Sidon property
        sums = []
        valid = True
        for i2 in range(len(U_copy)):
            for j2 in range(i2, len(U_copy)):
                s = U_copy[i2] + U_copy[j2]
                if s in sums:
                    valid = False
                    break
                sums.append(s)
            if not valid:
                break
        
        if valid:
            U = U_copy
    
    return sorted(U)

def _local_search(U, max_iter=1000, time_limit=10.0):
    """Local search to improve a set."""
    start_time = time.time()
    current_score = score_set(U)
    U = list(U)
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Try various mutations
        op = random.choice(['swap', 'add', 'remove', 'perturb'])
        new_U = U[:]
        
        if op == 'swap' and len(new_U) > 2:
            i = random.randint(1, len(new_U) - 1)  # Don't change 0
            new_val = random.randint(1, max(new_U) + 20)
            if new_val not in new_U:
                new_U[i] = new_val
                new_U.sort()
        elif op == 'add' and len(new_U) < 25:
            new_val = random.randint(1, max(new_U) + 30)
            if new_val not in new_U:
                new_U.append(new_val)
                new_U.sort()
        elif op == 'remove' and len(new_U) > 3:
            i = random.randint(1, len(new_U) - 1)
            new_U.pop(i)
            new_U.sort()
        elif op == 'perturb' and len(new_U) > 2:
            i = random.randint(1, len(new_U) - 1)
            delta = random.randint(-5, 5)
            new_val = new_U[i] + delta
            if new_val > 0 and new_val not in new_U:
                new_U[i] = new_val
                new_U.sort()
        
        if len(new_U) != len(U) or 0 not in new_U:
            continue
        
        new_score = score_set(new_U)
        if new_score > current_score:
            U = new_U
            current_score = new_score
    
    return U, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, 10s buffer
    
    best_score = 0.0
    best_set = [0, 1]
    
    # Try structured constructions
    for n in range(3, 20):
        if time.time() > deadline - 30:
            break
        
        # Golomb rulers
        try:
            g = _golomb_ruler(n)
            if 0 in g:
                s = score_set(g)
                if s > best_score:
                    best_score = s
                    best_set = g[:]
        except Exception:
            pass
        
        # Sidon sets
        try:
            sid = _sidon_ruler(n)
            if 0 in sid:
                s = score_set(sid)
                if s > best_score:
                    best_score = s
                    best_set = sid[:]
        except Exception:
            pass
        
        # Random sets
        random.seed(n * 1000)
        for _ in range(50):
            if time.time() > deadline - 20:
                break
            U = [0] + sorted(random.sample(range(1, n * 10), n - 1))
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Local search on best found
    if time.time() < deadline - 10:
        improved, improved_score = _local_search(best_set, max_iter=2000, time_limit=deadline - time.time() - 5)
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved
    
    # More random restarts if time permits
    random.seed(12345)
    while time.time() < deadline - 5:
        n = random.randint(3, 15)
        M = random.randint(n * 5, n * 20)
        U = [0] + sorted(random.sample(range(1, M), n - 1))
        
        # Quick local search
        U_ls, s_ls = _local_search(U, max_iter=100, time_limit=2.0)
        if s_ls > best_score:
            best_score = s_ls
            best_set = U_ls[:]
    
    return best_set
