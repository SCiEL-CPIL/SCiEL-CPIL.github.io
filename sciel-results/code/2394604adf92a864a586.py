import time
import math
import random
import numpy as np

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Find all divisors of 51975
    divisors = []
    for i in range(1, int(math.isqrt(base)) + 1):
        if base % i == 0:
            if i > 1:
                divisors.append(i)
            other = base // i
            if other > 1 and other != i:
                divisors.append(other)
    divisors.sort()
    
    # Precompute residue lists as numpy arrays for each (m, r)
    residues = {}
    for m in divisors:
        residues[m] = []
        for r in range(m):
            arr = np.arange(r, base, m, dtype=np.int32)
            residues[m].append(arr)
    
    # Greedy set cover with numpy acceleration
    def greedy_cover(seed_time):
        """Run greedy set cover, returning classes and freq array."""
        uncovered = np.ones(base, dtype=np.int8)
        freq = np.zeros(base, dtype=np.int32)
        used_moduli = set()
        classes = []
        remaining = set(divisors)
        
        while remaining and time.time() < seed_time:
            best_gain = 0
            best_m = None
            best_r = None
            
            for m in remaining:
                # Vectorized gain computation for all residues of modulus m
                # gain for (m, r) = sum of uncovered[x] for x in residues[m][r]
                for r in range(m):
                    gain = int(np.sum(uncovered[residues[m][r]]))
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = r
            
            if best_gain == 0:
                break
            
            classes.append([best_m, best_r])
            used_moduli.add(best_m)
            remaining.remove(best_m)
            
            arr = residues[best_m][best_r]
            freq[arr] += 1
            uncovered[arr] = np.where(freq[arr] > 0, 0, 1)
        
        return classes, freq, uncovered, used_moduli
    
    # Local search: add, remove, swap
    def local_search(classes, freq, uncovered, used_moduli, deadline):
        """Improve the solution via local search."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Add best unused class
            best_add_gain = 0
            best_add_m = None
            best_add_r = None
            for m in divisors:
                if m in used_moduli:
                    continue
                for r in range(m):
                    gain = int(np.sum(uncovered[residues[m][r]]))
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add_m = m
                        best_add_r = r
            
            if best_add_gain > 0:
                m, r = best_add_m, best_add_r
                classes.append([m, r])
                used_moduli.add(m)
                arr = residues[m][r]
                freq[arr] += 1
                uncovered[arr] = np.where(freq[arr] > 0, 0, 1)
                improved = True
                continue
            
            # Phase 2: Remove worst used class
            best_remove_loss = 0
            best_remove_idx = -1
            for idx, (m, r) in enumerate(classes):
                # loss = number of residues that would become uncovered
                loss = int(np.sum((freq[residues[m][r]] == 1)))
                if loss < best_remove_loss or best_remove_idx == -1:
                    if loss < best_remove_loss or best_remove_idx == -1:
                        best_remove_loss = loss
                        best_remove_idx = idx
            
            if best_remove_idx >= 0 and best_remove_loss == 0:
                m, r = classes[best_remove_idx]
                used_moduli.remove(m)
                arr = residues[m][r]
                freq[arr] -= 1
                uncovered[arr] = np.where(freq[arr] == 0, 1, 0)
                classes.pop(best_remove_idx)
                improved = True
                continue
            
            # Phase 3: Swap - try replacing one used class with one unused
            best_swap_gain = 0
            best_swap_old_idx = -1
            best_swap_new_m = None
            best_swap_new_r = None
            
            # Precompute losses for used classes
            used_losses = []
            for idx, (m, r) in enumerate(classes):
                loss = int(np.sum((freq[residues[m][r]] == 1)))
                used_losses.append((loss, idx, m, r))
            used_losses.sort()
            
            # Precompute gains for unused classes
            unused_gains = []
            for m in divisors:
                if m in used_moduli:
                    continue
                for r in range(m):
                    gain = int(np.sum(uncovered[residues[m][r]]))
                    if gain > 0:
                        unused_gains.append((gain, m, r))
            unused_gains.sort(reverse=True)
            
            # Try top candidates
            for g_new, m_new, r_new in unused_gains[:10]:
                for l_old, idx_old, m_old, r_old in used_losses[:10]:
                    net = g_new - l_old
                    if net > best_swap_gain:
                        best_swap_gain = net
                        best_swap_old_idx = idx_old
                        best_swap_new_m = m_new
                        best_swap_new_r = r_new
            
            if best_swap_gain > 0 and best_swap_old_idx >= 0:
                m_old, r_old = classes[best_swap_old_idx]
                # Remove old
                used_moduli.remove(m_old)
                arr_old = residues[m_old][r_old]
                freq[arr_old] -= 1
                uncovered[arr_old] = np.where(freq[arr_old] == 0, 1, 0)
                classes.pop(best_swap_old_idx)
                # Add new
                m_new, r_new = best_swap_new_m, best_swap_new_r
                classes.append([m_new, r_new])
                used_moduli.add(m_new)
                arr_new = residues[m_new][r_new]
                freq[arr_new] += 1
                uncovered[arr_new] = np.where(freq[arr_new] > 0, 0, 1)
                improved = True
        
        return classes, freq, uncovered, used_moduli
    
    # Multi-start: try several different greedy initializations
    best_classes = None
    best_uncovered_count = base
    
    # Start 1: Standard greedy
    classes1, freq1, unc1, used1 = greedy_cover(deadline - 30)
    classes1, freq1, unc1, used1 = local_search(classes1, freq1, unc1, used1, deadline - 10)
    unc_count = int(np.sum(unc1))
    if unc_count < best_uncovered_count:
        best_uncovered_count = unc_count
        best_classes = [c[:] for c in classes1]
    
    if time.time() > deadline - 20:
        return best_classes if best_classes else []
    
    # Start 2: Greedy with random tie-breaking
    random.seed(42)
    classes2, freq2, unc2, used2 = greedy_cover(deadline - 20)
    classes2, freq2, unc2, used2 = local_search(classes2, freq2, unc2, used2, deadline - 5)
    unc_count2 = int(np.sum(unc2))
    if unc_count2 < best_uncovered_count:
        best_uncovered_count = unc_count2
        best_classes = [c[:] for c in classes2]
    
    if time.time() > deadline - 5:
        return best_classes if best_classes else []
    
    # Start 3: Try a few more random perturbations of the best solution
    for attempt in range(3):
        if time.time() > deadline - 2:
            break
        
        # Perturb: randomly remove 1-2 classes and re-run local search
        perturbed = [c[:] for c in best_classes]
        freq_p = np.zeros(base, dtype=np.int32)
        used_p = set()
        for m, r in perturbed:
            freq_p[residues[m][r]] += 1
            used_p.add(m)
        
        # Remove 1 random class
        if perturbed:
            idx = random.randint(0, len(perturbed) - 1)
            m_rm, r_rm = perturbed[idx]
            freq_p[residues[m_rm][r_rm]] -= 1
            used_p.remove(m_rm)
            perturbed.pop(idx)
        
        unc_p = np.where(freq_p == 0, 1, 0).astype(np.int8)
        perturbed, freq_p, unc_p, used_p = local_search(perturbed, freq_p, unc_p, used_p, deadline - 1)
        unc_count_p = int(np.sum(unc_p))
        if unc_count_p < best_uncovered_count:
            best_uncovered_count = unc_count_p
            best_classes = [c[:] for c in perturbed]
    
    return best_classes if best_classes else []
