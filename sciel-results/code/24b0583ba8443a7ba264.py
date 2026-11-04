"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _word_to_int(word, q):
    """Convert a word (list of ints 0..q-1) to an integer index."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _int_to_word(idx, q, n):
    """Convert an integer index to a word (list of ints 0..q-1)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _generate_ball_indices(word, q, n, radius):
    """Generate all word indices within Hamming distance `radius` from `word`."""
    ball = {tuple(word)}
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(1, q), repeat=r):
                new_word = list(word)
                for pos, val in zip(positions, values):
                    new_word[pos] = (word[pos] + val) % q
                ball.add(tuple(new_word))
    return ball


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Leave safety margin before 150s kill
    
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    # Precompute ball index sets
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    ball_sets = []
    for w in all_words:
        ball = _generate_ball_indices(w, q, n, radius)
        ball_sets.append({word_to_idx[b] for b in ball})
    
    # Precompute ball size and pairwise overlap counts
    ball_size = np.array([len(bs) for bs in ball_sets], dtype=np.int32)
    # overlap[i][j] = |ball_i ∩ ball_j|
    # For radius=1, q=3, n=6: balls have size 1 + 6*2 = 13
    # overlap can be computed via: for each word j, count how many of its ball members are in ball_i
    # More efficient: overlap[i][j] = sum over positions of agreement * (q-1) adjustments
    # Actually simplest: precompute via set intersection (729*729 = 531k ops, fast enough)
    overlap = np.zeros((total_words, total_words), dtype=np.int32)
    for i in range(total_words):
        for j in range(total_words):
            overlap[i][j] = len(ball_sets[i] & ball_sets[j])
    
    # Full coverage mask
    full_mask = np.ones(total_words, dtype=bool)
    
    best_solution = None
    best_size = float('inf')
    
    def is_full_coverage(covered):
        """covered is a boolean array of shape (total_words,)"""
        return np.all(covered)
    
    def compute_gain(covered_bool):
        """For each word i, gain = |ball_i \\ covered| = ball_size[i] - |ball_i ∩ covered|
        |ball_i ∩ covered| = sum over j in ball_i of covered[j]
        = sum_j overlap[i][j] * covered[j]  (since overlap[i][j] = |ball_i ∩ ball_j| and ball_j contains j)
        Wait, that's not right. |ball_i ∩ covered| = number of j such that j in ball_i AND covered[j].
        = sum_j [j in ball_i] * covered[j]
        Let's precompute a matrix M where M[i][j] = 1 if j in ball_i, else 0.
        Then gain[i] = ball_size[i] - sum_j M[i][j] * covered[j]
        """
        pass  # We'll use the M matrix approach below
    
    # Precompute M: M[i][j] = 1 if j is in ball_i
    M = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        for j in ball_sets[i]:
            M[i][j] = 1
    
    # ball_size[i] = sum_j M[i][j]
    # gain[i] = ball_size[i] - M[i] @ covered_vector
    
    def greedy_set_cover():
        """Greedy set cover using precomputed matrices."""
        covered = np.zeros(total_words, dtype=bool)
        covered_vec = np.zeros(total_words, dtype=np.int32)
        solution = []
        
        while not np.all(covered):
            if time.time() > deadline - 20:
                break
            
            # gain[i] = ball_size[i] - M[i] @ covered_vec
            gains = ball_size - M @ covered_vec
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]
            
            if best_gain <= 0:
                break
            
            solution.append(best_idx)
            # Update covered
            new_members = M[best_idx] == 1
            covered[new_members] = True
            covered_vec[new_members] = 1
        
        return solution, covered
    
    def systematic_deletion(solution, covered):
        """Try removing each word; if coverage maintained, remove it."""
        changed = True
        while changed:
            changed = False
            for i in range(len(solution) - 1, -1, -1):
                if time.time() > deadline - 20:
                    return solution, covered
                # Check if removing solution[i] maintains coverage
                # New covered = covered minus (ball_{solution[i]} \ intersection with others)
                # Simpler: recompute from scratch if solution is small
                if len(solution) <= 80:
                    new_covered = np.zeros(total_words, dtype=bool)
                    for j, w_idx in enumerate(solution):
                        if j != i:
                            new_covered[M[w_idx] == 1] = True
                    if np.all(new_covered):
                        solution.pop(i)
                        covered = new_covered
                        changed = True
                        break
        return solution, covered
    
    def local_search_add_remove(solution, covered):
        """Local search: try removing a word, if not covered try adding a word to fix, then delete again."""
        temp = 5.0
        cooling = 0.95
        max_iters = 3000
        sol_set = set(solution)
        
        for sa_iter in range(max_iters):
            if time.time() > deadline - 20:
                break
            if len(solution) <= 1:
                break
            
            # Try removing a random codeword
            idx_to_remove = random.randint(0, len(solution) - 1)
            w_rm = solution[idx_to_remove]
            
            # Check coverage without this word
            # covered_without = covered minus (ball_{w_rm} \ union of other balls)
            # Fast check: a word j is only covered by w_rm if j in ball_{w_rm} and not in any other ball
            # j is uniquely covered by w_rm if: j in ball_{w_rm} AND for all other w_idx in solution, j not in ball_{w_idx}
            # i.e., j in ball_{w_rm} AND sum_{k != idx_to_remove} M[solution[k]][j] == 0
            
            # Compute: for each j in ball_{w_rm}, count how many other solution words cover j
            ball_rm = M[w_rm] == 1
            other_cover_count = np.zeros(total_words, dtype=np.int32)
            for k, w_idx in enumerate(solution):
                if k != idx_to_remove:
                    other_cover_count[M[w_idx] == 1] += 1
            
            # Words uniquely covered by w_rm:
            unique = ball_rm & (other_cover_count == 0)
            n_unique = np.sum(unique)
            
            if n_unique == 0:
                # Safe to remove
                solution.pop(idx_to_remove)
                sol_set.discard(w_rm)
                # Update covered
                covered[ball_rm & (other_cover_count == 0)] = False
                # Actually covered should still be True for those if others cover them
                # Recompute covered properly
                covered = np.zeros(total_words, dtype=bool)
                for w_idx in solution:
                    covered[M[w_idx] == 1] = True
                temp *= cooling
            else:
                # Not safe. Try to find a word to add that covers the unique words.
                # Accept removal with some probability if penalty is small (SA)
                if random.random() < np.exp(-n_unique / max(temp, 0.1)):
                    # Try to fix by adding a word
                    # Find word that covers most of the unique words
                    unique_indices = np.where(unique)[0]
                    if len(unique_indices) > 0:
                        # For each candidate word, count how many unique words it covers
                        # candidate covers unique word j if M[candidate][j] == 1
                        # = M[:, unique_indices] summed over columns
                        cov_counts = M[:, unique_indices].sum(axis=1)
                        # Exclude words already in solution
                        for w_idx in solution:
                            cov_counts[w_idx] = 0
                        best_add = int(np.argmax(cov_counts))
                        if cov_counts[best_add] > 0:
                            # Add best_add, remove w_rm
                            solution[idx_to_remove] = best_add
                            sol_set.discard(w_rm)
                            sol_set.add(best_add)
                            # Update covered
                            covered = np.zeros(total_words, dtype=bool)
                            for w_idx in solution:
                                covered[M[w_idx] == 1] = True
                            temp *= cooling
                            continue
                temp *= cooling
        
        return solution, covered
    
    # Try multiple random restarts
    for attempt in range(30):
        if time.time() > deadline - 25:
            break
        
        random.seed(42 + attempt)
        
        # Phase 1: Greedy set cover
        solution, covered = greedy_set_cover()
        
        if not np.all(covered):
            continue
        
        # Phase 2: Systematic deletion
        solution, covered = systematic_deletion(solution, covered)
        
        # Phase 3: Local search with add-remove
        solution, covered = local_search_add_remove(solution, covered)
        
        # Phase 4: Another systematic deletion pass
        solution, covered = systematic_deletion(solution, covered)
        
        # Validate and update best
        if np.all(covered):
            # Verify no duplicates
            sol_set = set(solution)
            if len(sol_set) == len(solution):
                if len(solution) < best_size:
                    best_size = len(solution)
                    best_solution = [list(_int_to_word(w, q, n)) for w in solution]
    
    if best_solution is None:
        # Fallback: use direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return best_solution
# EVOLVE-BLOCK-END
