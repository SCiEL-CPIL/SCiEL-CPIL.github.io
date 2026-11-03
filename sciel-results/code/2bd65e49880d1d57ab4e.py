"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import permutations

# EVOLVE-BLOCK-START

def _classic_superperm(n):
    """Build the classic superpermutation of length sum(k! for k=1..n)."""
    word = [1]
    for m in range(2, n + 1):
        # Collect all (m-1)-permutations in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between end of new and start of block
            overlap = 0
            max_ov = min(len(new), len(block) - 1)
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _is_valid_superperm(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive windows."""
    expected = math.factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            seen.add(w)
            if len(seen) == expected:
                return True
    return len(seen) == expected

class _SuperpermState:
    """
    Maintains the count of occurrences of each permutation in a word.
    Allows efficient checking and updating for deletions.
    """
    def __init__(self, word, n):
        self.n = n
        self.word = word[:]
        self.length = len(self.word)
        self.perm_counts = {}
        self.total_unique = 0
        self.expected = math.factorial(n)
        
        # Initialize counts
        for i in range(self.length - n + 1):
            w = tuple(self.word[i:i+n])
            if len(set(w)) == n:
                if w in self.perm_counts:
                    self.perm_counts[w] += 1
                else:
                    self.perm_counts[w] = 1
                    self.total_unique += 1
                    
    def is_valid(self):
        return self.total_unique == self.expected

    def try_delete(self, i):
        """
        Try to delete element at index i.
        Returns True if successful, False otherwise.
        """
        n = self.n
        L = self.length
        
        if i <= 0 or i >= L - 1:
            return False
            
        # Windows that will be removed (start indices k such that window k..k+n-1 contains i)
        # i is in window starting at k if k <= i <= k+n-1  =>  i-n+1 <= k <= i
        # Also k must be valid start: 0 <= k <= L-n
        start_min = max(0, i - n + 1)
        start_max = min(i, L - n)
        
        removed_windows = []
        for k in range(start_min, start_max + 1):
            w = tuple(self.word[k:k+n])
            if len(set(w)) == n:
                removed_windows.append(w)
                # Check if this is the last occurrence
                if self.perm_counts[w] == 1:
                    # Cannot delete, would lose this permutation
                    return False
        
        # If we reach here, deletion is safe.
        # Perform deletion
        new_word = self.word[:i] + self.word[i+1:]
        
        # Update counts
        # 1. Decrement counts for removed windows
        for w in removed_windows:
            self.perm_counts[w] -= 1
            if self.perm_counts[w] == 0:
                del self.perm_counts[w]
                self.total_unique -= 1
        
        # 2. Add new windows formed by bridging the gap
        # New windows start at k in [i-n+1, i-1] (relative to new word indices? No, let's trace carefully)
        # Old word indices: ... i-1, i, i+1 ...
        # New word indices: ... i-1, i+1 ... (i+1 becomes i)
        # A new window in new word starting at k_new consists of:
        # new_word[k_new : k_new+n]
        # If k_new < i, the window might span the deletion point.
        # The windows that change are those that include the position where i was.
        # In the new word, the gap is at index i (elements after i shift left).
        # Windows in new word starting at k_new such that they cover the "seam" at i.
        # The seam is between new_word[i-1] and new_word[i] (which is old_word[i+1]).
        # So windows starting at k_new where k_new <= i-1 < k_new+n.
        # i.e., i-n+1 <= k_new <= i-1.
        # Also k_new must be valid: 0 <= k_new <= new_len - n.
        
        new_len = L - 1
        new_start_min = max(0, i - n + 1)
        new_start_max = min(i - 1, new_len - n)
        
        for k_new in range(new_start_min, new_start_max + 1):
            # Construct the new window
            # It consists of new_word[k_new : i] + new_word[i : k_new+n]
            # Which is old_word[k_new : i] + old_word[i+1 : k_new+n+1]
            left_part = self.word[k_new:i]
            right_part = self.word[i+1:k_new+n+1]
            w_new = tuple(left_part + right_part)
            
            if len(set(w_new)) == n:
                if w_new in self.perm_counts:
                    self.perm_counts[w_new] += 1
                else:
                    self.perm_counts[w_new] = 1
                    self.total_unique += 1
        
        self.word = new_word
        self.length = new_len
        return True

def _score_deletion_candidates(state, indices):
    """
    Score deletion candidates by how 'redundant' they are.
    A position is more redundant if it appears in windows with higher counts.
    Returns list of (score, index) sorted by score descending.
    """
    scored = []
    for i in indices:
        if i <= 0 or i >= state.length - 1:
            continue
        n = state.n
        L = state.length
        start_min = max(0, i - n + 1)
        start_max = min(i, L - n)
        
        # Score = sum of counts of windows containing this position
        # Higher count means more redundant (safer to delete)
        score = 0
        for k in range(start_min, start_max + 1):
            w = tuple(state.word[k:k+n])
            if len(set(w)) == n:
                score += state.perm_counts.get(w, 0)
        
        scored.append((score, i))
    
    # Sort by score descending (most redundant first)
    scored.sort(key=lambda x: -x[0])
    return scored

def construct(n):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s kill
    
    # For n <= 4, classic construction is optimal
    if n <= 4:
        word = _classic_superperm(n)
        return "".join(map(str, word))
    
    # Build classic superpermutation as baseline
    best_word = _classic_superperm(n)
    best_len = len(best_word)
    
    # Verify baseline
    if not _is_valid_superperm(best_word, n):
        # Should never happen, but safety
        return "".join(map(str, best_word))
    
    # Use incremental local search for n >= 5
    if n >= 5:
        state = _SuperpermState(best_word, n)
        
        # Phase 1: Scored greedy deletion - prioritize most redundant positions
        # This should find more deletions than random order
        while time.time() < deadline - 40.0:
            improved = False
            
            # Get scored candidates
            indices = list(range(1, state.length - 1))
            scored = _score_deletion_candidates(state, indices)
            
            # Try deletions in order of redundancy
            for score, i in scored:
                if time.time() > deadline - 40.0:
                    break
                if state.try_delete(i):
                    improved = True
            
            if not improved:
                break
        
        # Phase 2: Multi-start random restarts to escape local minima
        rng = random.Random(42)
        best_local_len = state.length
        
        for attempt in range(30):
            if time.time() > deadline - 10.0:
                break
            
            # Save current state
            saved_word = state.word[:]
            saved_counts = dict(state.perm_counts)
            saved_total = state.total_unique
            saved_len = state.length
            
            # Try random deletions with occasional scored reordering
            found_deletion = False
            num_deletions = 0
            
            # Mix of random and scored approaches
            use_scored = (attempt % 3 == 0)
            
            if use_scored:
                indices = list(range(1, state.length - 1))
                scored = _score_deletion_candidates(state, indices)
                candidate_list = [i for _, i in scored]
            else:
                candidate_list = list(range(1, state.length - 1))
                rng.shuffle(candidate_list)
            
            for i in candidate_list:
                if time.time() > deadline - 10.0:
                    break
                if i >= state.length - 1 or i <= 0:
                    continue
                if state.try_delete(i):
                    found_deletion = True
                    num_deletions += 1
                    
                    # After some deletions, re-score to adapt
                    if num_deletions % 20 == 0 and num_deletions > 0:
                        remaining = [j for j in range(1, state.length - 1) if j not in candidate_list[:candidate_list.index(i)+1]]
                        if remaining:
                            if use_scored:
                                scored_rem = _score_deletion_candidates(state, remaining)
                                candidate_list = candidate_list[:candidate_list.index(i)+1] + [i2 for _, i2 in scored_rem]
                            else:
                                rng.shuffle(candidate_list)
            
            # Keep if we found improvements
            if found_deletion and len(state.word) < best_local_len:
                best_local_len = len(state.word)
                # Don't restore - keep the improvement
            else:
                # Restore if no improvement
                state.word = saved_word
                state.perm_counts = saved_counts
                state.total_unique = saved_total
                state.length = saved_len
        
        # Phase 3: Final polish with multiple passes
        for _ in range(30):
            if time.time() > deadline - 2.0:
                break
            improved = False
            for i in range(state.length - 2, 0, -1):
                if time.time() > deadline - 2.0:
                    break
                if state.try_delete(i):
                    improved = True
            if not improved:
                break
        
        if state.is_valid() and len(state.word) < best_len:
            best_word = state.word
            best_len = len(best_word)
    
    return "".join(map(str, best_word))

# EVOLVE-BLOCK-END
