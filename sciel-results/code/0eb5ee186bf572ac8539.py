import time
import random
import sys

def _check_sidon_unique_sums(els):
    """Check that all pair sums a+b (a<=b) are unique."""
    sums = set()
    n = len(els)
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_nonzero_diffs(els):
    """Get set of nonzero differences (positive and negative)."""
    diffs = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = els[j] - els[i]
                if d != 0:
                    diffs.add(d)
    return diffs

def _is_valid_pair(A, B):
    """Check that A and B are Sidon and their nonzero difference sets are disjoint."""
    if not A or not B:
        return False
    if not _check_sidon_unique_sums(A):
        return False
    if not _check_sidon_unique_sums(B):
        return False
    diffsA = _get_nonzero_diffs(A)
    diffsB = _get_nonzero_diffs(B)
    if diffsA & diffsB:
        return False
    return True

class SidonBuilder:
    def __init__(self, n=256, max_size=64):
        self.n = n
        self.max_size = max_size
        
    def _can_add_to_sidon(self, x, current_set, used_sums):
        """Check if x can be added to current_set maintaining Sidon property."""
        # Check sums with existing elements
        for a in current_set:
            s = a + x
            if s in used_sums:
                return False
        # Check diagonal sum
        if 2 * x in used_sums:
            return False
        return True

    def _diffs_with(self, x, current_set):
        """Get differences between x and all elements in current_set."""
        return set(x - a for a in current_set) | set(a - x for a in current_set)

    def greedy_build(self, fixed_set, used_diffs_fixed, rng, order='ascending'):
        """
        Greedily build a set disjoint in differences from fixed_set.
        fixed_set: The other set (A if building B, or B if building A).
        used_diffs_fixed: Differences of the fixed set.
        """
        new_set = []
        used_sums = set()
        
        candidates = list(range(1, self.n + 1))
        if order == 'descending':
            candidates.sort(reverse=True)
        elif order == 'random':
            rng.shuffle(candidates)
        else:
            candidates.sort()
            
        for x in candidates:
            if len(new_set) >= self.max_size:
                break
            
            # 1. Check Sidon property internally
            if not self._can_add_to_sidon(x, new_set, used_sums):
                continue
                
            # 2. Check difference disjointness with fixed_set
            # Differences of x with new_set elements must not conflict with used_diffs_fixed
            # AND differences of x with new_set elements must not conflict with each other?
            # No, Sidon property on sums ensures differences within new_set are unique?
            # Wait. Sidon property on sums {a_i + a_j} implies that differences {a_i - a_j} are unique?
            # If a_i - a_j = a_k - a_l, then a_i + a_l = a_k + a_j.
            # If i!=j and k!=l, and the pairs {i,l} and {k,j} are distinct, this is a sum collision.
            # So yes, Sidon sets have unique nonzero differences.
            # So we just need to ensure:
            # a) New sums don't collide with existing sums in new_set.
            # b) New differences (x - a for a in new_set) don't collide with used_diffs_fixed.
            # c) Also, differences (x - a) must not collide with differences already in new_set?
            #    Since new_set is Sidon, its differences are unique.
            #    We must ensure x - a is not in diffs(new_set).
            
            new_diffs = self._diffs_with(x, new_set)
            
            # Check against fixed set diffs
            if new_diffs & used_diffs_fixed:
                continue
                
            # Check against new_set's own diffs
            # We need to compute diffs of new_set.
            # To speed this up, we can maintain a set of diffs for new_set.
            # But for simplicity in this helper, let's compute or maintain.
            # Let's maintain used_diffs_new.
            
            # For now, just check if any new_diff is in the set of diffs of new_set.
            # This is O(k^2) if done naively.
            # Let's assume we maintain used_diffs_new in the main loop.
            
            # Let's refine the greedy loop to maintain used_diffs_new.
            
            # Re-implementing logic inside the loop for efficiency is better.
            # This helper is just a sketch.
            
            # Add to set
            new_set.append(x)
            # Update used_sums
            for a in new_set[:-1]:
                used_sums.add(a + x)
            used_sums.add(2 * x)
            
        return new_set

    def build_pair(self, rng, strategy='asc'):
        """
        Build a pair (A, B).
        Strategy:
        1. Start with empty A and B.
        2. Alternately add elements to A and B, or build one then the other.
        Let's try: Build A greedily. Then Build B greedily avoiding A's diffs.
        Then try to add more to A avoiding B's diffs.
        """
        A = []
        B = []
        
        used_sums_A = set()
        used_sums_B = set()
        used_diffs_A = set()
        used_diffs_B = set()
        
        # Helper to add x to a set
        def try_add(x, target_list, used_sums, used_diffs_self, used_diffs_other):
            # Check Sidon sums
            for a in target_list:
                if a + x in used_sums:
                    return False
            if 2 * x in used_sums:
                return False
            
            # Check Diffs
            new_diffs = set()
            for a in target_list:
                d1 = x - a
                d2 = a - x
                new_diffs.add(d1)
                new_diffs.add(d2)
                
            # Must not conflict with other set's diffs
            if new_diffs & used_diffs_other:
                return False
                
            # Must not conflict with self's existing diffs
            if new_diffs & used_diffs_self:
                return False
                
            # Commit
            target_list.append(x)
            for a in target_list[:-1]:
                used_sums.add(a + x)
            used_sums.add(2 * x)
            used_diffs_self.update(new_diffs)
            return True

        # Phase 1: Build A
        candidates = list(range(1, self.n + 1))
        if strategy == 'desc':
            candidates.sort(reverse=True)
        elif strategy == 'rand':
            rng.shuffle(candidates)
        else:
            candidates.sort()
            
        for x in candidates:
            if len(A) >= self.max_size:
                break
            if try_add(x, A, used_sums_A, used_diffs_A, used_diffs_B):
                pass

        # Phase 2: Build B
        candidates_b = list(range(1, self.n + 1))
        if strategy == 'desc':
            candidates_b.sort(reverse=True)
        elif strategy == 'rand':
            rng.shuffle(candidates_b)
        else:
            candidates_b.sort()
            
        for x in candidates_b:
            if len(B) >= self.max_size:
                break
            if try_add(x, B, used_sums_B, used_diffs_B, used_diffs_A):
                pass

        # Phase 3: Try to extend A again with B's diffs now fixed
        candidates_a2 = list(range(1, self.n + 1))
        if strategy == 'desc':
            candidates_a2.sort(reverse=True)
        elif strategy == 'rand':
            rng.shuffle(candidates_a2)
        else:
            candidates_a2.sort()
            
        for x in candidates_a2:
            if len(A) >= self.max_size:
                break
            if x in A:
                continue
            if try_add(x, A, used_sums_A, used_diffs_A, used_diffs_B):
                pass

        # Phase 4: Try to extend B again
        candidates_b2 = list(range(1, self.n + 1))
        if strategy == 'desc':
            candidates_b2.sort(reverse=True)
        elif strategy == 'rand':
            rng.shuffle(candidates_b2)
        else:
            candidates_b2.sort()
            
        for x in candidates_b2:
            if len(B) >= self.max_size:
                break
            if x in B:
                continue
            if try_add(x, B, used_sums_B, used_diffs_B, used_diffs_A):
                pass

        return A, B

def _local_search_improve(A, B, n=256, max_iters=50, rng=None):
    """
    Improve by swapping elements.
    """
    if rng is None:
        rng = random.Random()
    
    best_A = list(A)
    best_B = list(B)
    best_size = len(A) + len(B)
    
    for _ in range(max_iters):
        improved = False
        
        # Try to add to A
        for _ in range(10):
            x = rng.randint(1, n)
            if x in best_A:
                continue
            new_A = best_A + [x]
            if _is_valid_pair(new_A, best_B):
                best_A = new_A
                best_size = len(best_A) + len(best_B)
                improved = True
                break
                
        # Try to add to B
        for _ in range(10):
            x = rng.randint(1, n)
            if x in best_B:
                continue
            new_B = best_B + [x]
            if _is_valid_pair(best_A, new_B):
                best_B = new_B
                best_size = len(best_A) + len(best_B)
                improved = True
                break
                
        # Try swap in A
        if len(best_A) > 0:
            idx = rng.randint(0, len(best_A) - 1)
            old_x = best_A[idx]
            for _ in range(10):
                new_x = rng.randint(1, n)
                if new_x in best_A:
                    continue
                new_A = best_A[:idx] + [new_x] + best_A[idx+1:]
                if _is_valid_pair(new_A, best_B):
                    best_A = new_A
                    improved = True
                    break
                    
        # Try swap in B
        if len(best_B) > 0:
            idx = rng.randint(0, len(best_B) - 1)
            old_x = best_B[idx]
            for _ in range(10):
                new_x = rng.randint(1, n)
                if new_x in best_B:
                    continue
                new_B = best_B[:idx] + [new_x] + best_B[idx+1:]
                if _is_valid_pair(best_A, new_B):
                    best_B = new_B
                    improved = True
                    break

        if not improved:
            break
            
    return best_A, best_B

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 256
    max_size = 64
    
    best_A = [1]
    best_B = [2]
    best_score = 2
    
    rng = random.Random()
    builder = SidonBuilder(n, max_size)
    
    strategies = ['asc', 'desc', 'rand', 'rand', 'rand']
    
    for restart in range(10000):
        if time.time() > deadline:
            break
            
        strategy = strategies[restart % len(strategies)]
        A, B = builder.build_pair(rng, strategy)
        
        if not A: A = [1]
        if not B: B = [2]
        
        if _is_valid_pair(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
                
        # Local search
        if time.time() < deadline - 5:
            A_ls, B_ls = _local_search_improve(A, B, n, max_iters=20, rng=rng)
            if _is_valid_pair(A_ls, B_ls):
                score_ls = len(A_ls) + len(B_ls)
                if score_ls > best_score:
                    best_score = score_ls
                    best_A = list(A_ls)
                    best_B = list(B_ls)
                    
    if not _is_valid_pair(best_A, best_B):
        best_A = [1]
        best_B = [2]
        
    return [best_A, best_B]
