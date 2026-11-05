import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s evaluator
    
    # Total number of words
    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # For radius=1: 1 + n*(q-1)
    
    # Precompute balls as a 2D numpy array: balls[i] = list of indices in ball around i
    balls = np.zeros((total_words, ball_size), dtype=np.int32)
    
    for i in range(total_words):
        # Convert index to word
        word = []
        tmp = i
        for _ in range(n):
            word.append(tmp % q)
            tmp //= q
        word = word[::-1]  # Reverse to get correct order
        
        # Ball contains i itself
        balls[i, 0] = i
        
        # Generate all neighbors at distance 1
        idx = 1
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_word = word.copy()
                new_word[pos] = new_val
                # Convert back to index
                new_idx = 0
                for k in range(n):
                    new_idx = new_idx * q + new_word[k]
                balls[i, idx] = new_idx
                idx += 1
    
    # Precompute neighbor masks for vectorized gain computation
    # For each codeword position c, we need to know which words it covers
    # Use a sparse representation: for each word w, which codewords cover it?
    # Actually, let's use a different approach: precompute for each codeword c,
    # the set of words it covers as a boolean mask would be too large.
    # Instead, we'll compute gain by checking which words in balls[c] are uncovered.
    
    def greedy_cover(covered_mask, max_time=None):
        """Greedy set cover using vectorized gain computation."""
        if max_time is None:
            max_time = time_limit - (time.time() - start_time)
        
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)
        covered = covered_mask.copy()
        
        while not covered.all():
            if time.time() - start_time > time_limit:
                break
            
            # Compute gain for all candidates: how many uncovered words each covers
            # For each candidate c, gain[c] = sum of (1 - covered[j]) for j in balls[c]
            # Vectorized: for each row c in balls, sum (1-covered) over that row
            gains = np.sum(1 - covered[balls], axis=1)
            
            # Find best candidate
            best_idx = int(np.argmax(gains))
            best_gain = int(gains[best_idx])
            
            if best_gain == 0:
                break
            
            # Add this codeword
            code.append(best_idx)
            cover_count[balls[best_idx]] += 1
            covered[balls[best_idx]] = True
        
        return code, cover_count, covered
    
    def local_search(code, cover_count, covered):
        """Local search: try removals and swaps."""
        code = list(code)
        cover_count = cover_count.copy()
        covered = covered.copy()
        
        def is_redundant(idx):
            # Check if all words in ball[idx] have cover_count > 1
            return np.all(cover_count[balls[idx]] > 1)
        
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            
            # Try removals
            for i in range(len(code)):
                if is_redundant(code[i]):
                    # Remove codeword i
                    for j in balls[code[i]]:
                        cover_count[j] -= 1
                        if cover_count[j] == 0:
                            covered[j] = False
                    code.pop(i)
                    improved = True
                    break
            
            if not improved and len(code) > 0:
                # Try a kick: remove one codeword, greedily repair
                kick_pos = random.randint(0, len(code) - 1)
                kick_idx = code[kick_pos]
                
                old_code = code.copy()
                old_cover_count = cover_count.copy()
                old_covered = covered.copy()
                
                # Remove the kicked codeword
                for j in balls[kick_idx]:
                    cover_count[j] -= 1
                    if cover_count[j] == 0:
                        covered[j] = False
                code.pop(kick_pos)
                
                # Greedily repair
                while not covered.all():
                    if time.time() - start_time > time_limit:
                        break
                    
                    gains = np.sum(1 - covered[balls], axis=1)
                    best_idx = int(np.argmax(gains))
                    best_gain = int(gains[best_idx])
                    
                    if best_gain == 0:
                        break
                    
                    code.append(best_idx)
                    cover_count[balls[best_idx]] += 1
                    covered[balls[best_idx]] = True
                
                # Accept if improved
                if len(code) < len(old_code):
                    improved = True
                else:
                    # Revert
                    code = old_code
                    cover_count = old_cover_count
                    covered = old_covered
        
        return code, cover_count, covered
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Try multiple restarts with different random seeds
    best_code = None
    best_len = float('inf')
    
    num_restarts = 5
    per_restart_time = (time_limit - (time.time() - start_time)) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Set random seed for reproducibility within restart
        random.seed(42 + restart)
        
        # Greedy phase
        initial_covered = np.zeros(total_words, dtype=bool)
        code, cover_count, covered = greedy_cover(initial_covered, max_time=per_restart_time * 0.3)
        
        if time.time() - start_time > time_limit:
            break
        
        # Local search
        code, cover_count, covered = local_search(code, cover_count, covered)
        
        # Track best
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
        
        # If we've reached a good solution, spend remaining time on final polish
        if best_len <= 190:
            break
    
    if best_code is None:
        # Fallback: run greedy once more
        initial_covered = np.zeros(total_words, dtype=bool)
        best_code, _, _ = greedy_cover(initial_covered)
    
    # Final cleanup: try more removals
    if time.time() - start_time < time_limit - 5:
        cover_count = np.zeros(total_words, dtype=np.int32)
        covered = np.zeros(total_words, dtype=bool)
        for idx in best_code:
            cover_count[balls[idx]] += 1
            covered[balls[idx]] = True
        
        def is_redundant(idx):
            return np.all(cover_count[balls[idx]] > 1)
        
        removed = True
        while removed and (time.time() - start_time < time_limit - 5):
            removed = False
            for i in range(len(best_code)):
                if is_redundant(best_code[i]):
                    for j in balls[best_code[i]]:
                        cover_count[j] -= 1
                        if cover_count[j] == 0:
                            covered[j] = False
                    best_code.pop(i)
                    removed = True
                    break
    
    return [idx_to_word(idx) for idx in best_code]
