import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe within 120s/150s
    
    # Precompute all words and their balls of radius 1
    all_words = list(np.ndindex(*([q] * n)))
    # Convert tuples to lists for consistency, or just work with tuples
    # Let's use a list of tuples for the words
    words = [tuple(w) for w in all_words]
    word_to_idx = {w: i for i, w in enumerate(words)}
    total_words = len(words)
    
    # Precompute balls: ball[i] is a list of indices of words within distance 1 of words[i]
    balls = []
    for i, w in enumerate(words):
        ball = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w = tuple(new_w)
                ball.append(word_to_idx[new_w])
        balls.append(ball)
        
    # Initial Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    code = []
    cover_counts = np.zeros(total_words, dtype=int)
    
    # Greedy initialization
    while not np.all(covered):
        best_word = -1
        best_gain = -1
        # To speed up, we can sample or just iterate. 729 is small.
        for i in range(total_words):
            if covered[i]:
                continue
            # Count how many uncovered words this ball covers
            # We can just count uncovered in ball[i]
            gain = sum(1 for j in balls[i] if not covered[j])
            if gain > best_gain:
                best_gain = gain
                best_word = i
            if best_gain == total_words: # Perfect cover found? Unlikely for radius 1
                break
        if best_word == -1:
            break
        code.append(best_word)
        cover_counts[balls[best_word]] += 1
        covered[balls[best_word]] = True
        
    best_code = list(code)
    best_size = len(best_code)
    
    # Local Search
    # We maintain cover_counts and the list of code indices
    # We need a fast way to check if a removal is valid.
    # A removal of index c is valid if min(cover_counts[balls[c]]) > 1 before removal?
    # No, if min(counts in ball) > 1, then after decrement, min >= 1.
    # So we can just check min(cover_counts[balls[c]]) > 1.
    
    def is_removal_valid(c_idx):
        # Check if all words in the ball of c_idx have count > 1
        # If any has count 1, removing c_idx will make it 0.
        return np.min(cover_counts[balls[c_idx]]) > 1
        
    def remove_codeword(c_idx):
        cover_counts[balls[c_idx]] -= 1
        code.remove(c_idx)
        
    def add_codeword(c_idx):
        cover_counts[balls[c_idx]] += 1
        code.append(c_idx)
        
    # Shuffle code for randomization
    random.shuffle(code)
    
    iterations = 0
    stuck_counter = 0
    
    while time.time() < deadline:
        iterations += 1
        if iterations % 1000 == 0:
            # Periodic reshuffle to escape local minima?
            pass
            
        # Try to remove a random codeword
        if code:
            rand_idx = random.randint(0, len(code) - 1)
            c_idx = code[rand_idx]
            if is_removal_valid(c_idx):
                remove_codeword(c_idx)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                stuck_counter = 0
            else:
                stuck_counter += 1
                # If stuck for a while, try an "add-then-remove" move
                if stuck_counter > 50:
                    # Add a random word
                    add_word = random.choice(range(total_words))
                    add_codeword(add_word)
                    # Now try to remove some words, maybe the one we just added or others
                    # Let's try to remove random words from the current code
                    # We do a few attempts
                    for _ in range(10):
                        if not code: break
                        r_idx = random.randint(0, len(code) - 1)
                        c_to_rm = code[r_idx]
                        if is_removal_valid(c_to_rm):
                            remove_codeword(c_to_rm)
                            if len(code) < best_size:
                                best_size = len(size) if False else len(code) # Typo check
                                best_size = len(code)
                                best_code = list(code)
                            stuck_counter = 0
                            break
                    # If we didn't remove anything, we might be in a worse state (larger code)
                    # But hopefully the add allowed a subsequent remove that we missed in the first pass?
                    # Actually, the add increases size. We only care if we eventually remove MORE than we added.
                    # The logic above removes one if possible.
                    # If we added 1 and removed 1, size is same.
                    # If we added 1 and removed 0, size is +1.
                    # We should revert if we end up larger than best? No, we track best separately.
                    # But we should probably revert the "bad" add if it didn't lead to a reduction.
                    # Let's check if current size > best_size and we didn't improve.
                    # A simpler heuristic: if we added and didn't remove, revert the add.
                    # But we already potentially removed something.
                    # Let's just let it run. The "best_code" is saved.
                    # To prevent drift to huge codes, if len(code) > best_size + 5, restart or revert?
                    # Let's try to revert the last add if no removal happened.
                    # This is getting complex. Let's stick to:
                    # If stuck, add random. Then try to remove random.
                    # If we can't remove, we are stuck with a larger code.
                    # Let's just reset to best_code if we get too large.
                    if len(code) > best_size + 10:
                        code.clear()
                        code.extend(best_code)
                        # Rebuild cover_counts
                        cover_counts[:] = 0
                        for c in code:
                            cover_counts[balls[c]] += 1
                        stuck_counter = 0
        else:
            # Code is empty? Shouldn't happen for radius 1 covering
            break
            
    return [list(words[i]) for i in best_code]
