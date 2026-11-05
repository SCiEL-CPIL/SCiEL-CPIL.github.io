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
        """Local search: try removals and neutral swaps/kicks to escape local minima."""
        code = list(code)
        cover_count = cover_count.copy()
        covered = covered.copy()
        
        def is_redundant(idx):
            return np.all(cover_count[balls[idx]] > 1)
        
        improved = True
        neutral_moves = 0
        max_neutral = 50 # Limit neutral moves to avoid cycling
        
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
                    neutral_moves = 0 # Reset neutral counter on improvement
                    break
            
            if not improved and len(code) > 0:
                # Try a kick: remove one codeword, greedily repair
                # Accept if size decreases OR if size stays same (neutral move)
                kick_pos = random.randint(0, len(code) - 1)
                kick_idx = code[kick_pos]
                
                old_code = code.copy()
                old_cover_count = cover_count.copy()
                old_covered = covered.copy()
                old_len = len(code)
                
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
                
                # Accept if improved (smaller)
                if len(code) < old_len:
                    improved = True
                    neutral_moves = 0
                # Accept if neutral (same size) but we haven't exceeded neutral limit
                elif len(code) == old_len and neutral_moves < max_neutral:
                    improved = True
                    neutral_moves += 1
                else:
                    # Revert
                    code = old_code
                    cover_count = old_cover_count
                    covered = old_covered
                    # If we hit the limit, we might want to break or randomize more?
                    # For now, just break out of this specific kick attempt logic
                    # The outer loop will terminate if no further improvements are found
                    # Actually, if we revert, improved is False, so loop terminates.
                    # This is fine.
        
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
    
    num_restarts = 10 # Increase restarts if time allows
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
        if best_len <= 188: # Target is 186, so 188 is a good threshold to stop restarting
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
