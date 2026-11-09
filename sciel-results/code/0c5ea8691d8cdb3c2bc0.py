import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    # Generate all words
    all_words = np.array([list(w) for w in __import__('itertools').product(range(q), repeat=n)])
    num_words = q ** n
    
    # Precompute: for each word, the set of words within Hamming distance <= radius
    # For radius=1, n=7, q=3: each word covers itself + 7*2 = 15 words
    
    # Build ball membership: for each word, list of words it covers
    # Use a bitset approach for efficiency
    # Map each word to an index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
    
    # Precompute balls
    balls = []
    for i in range(num_words):
        w = all_words[i]
        covered = set([i])
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                covered.add(word_to_idx[tuple(w)])
                w[pos] = orig
        balls.append(covered)
    
    # Greedy set cover
    uncovered = set(range(num_words))
    code = []
    
    while uncovered:
        if time.time() > deadline - 5:
            break
        best_idx = -1
        best_count = 0
        for i in range(num_words):
            if i in code_set if 'code_set' in dir() else True:
                cnt = len(balls[i] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
        if best_idx == -1 or best_count == 0:
            break
        code.append(best_idx)
        uncovered -= balls[best_idx]
    
    code_set = set(code)
    
    # Try to remove redundant codewords
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            # Check if removing code[i] still leaves everything covered
            test_set = set(code[:i] + code[i+1:])
            # Check coverage
            covered = set()
            for idx in test_set:
                covered |= balls[idx]
            if covered == set(range(num_words)):
                code.pop(i)
                code_set = set(code)
                changed = True
                break
    
    # Try to improve by replacing two codewords with one
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for i in range(len(code)):
            if time.time() > deadline - 15:
                break
            for j in range(i+1, len(code)):
                if time.time() > deadline - 15:
                    break
                # Try removing both i and j, then greedily add best single codeword
                test_set = set(code[:i] + code[i+1:j] + code[j+1:])
                # Find uncovered
                covered = set()
                for idx in test_set:
                    covered |= balls[idx]
                uncovered = set(range(num_words)) - covered
                if not uncovered:
                    # Both are redundant
                    code = [c for c in code if c not in (code[i], code[j])]
                    code_set = set(code)
                    improved = True
                    break
                # Find best single codeword to cover remaining
                best_idx = -1
                best_count = 0
                for k in range(num_words):
                    if k in test_set:
                        continue
                    cnt = len(balls[k] & uncovered)
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = k
                if best_count >= len(uncovered):
                    # Replace both with one
                    code = [c for c in code if c not in (code[i], code[j])]
                    code.append(best_idx)
                    code_set = set(code)
                    improved = True
                    break
            if improved:
                break
    
    # Convert back to list of lists
    result = [all_words[i].tolist() for i in code]
    
    # Verify validity
    if time.time() < deadline - 5:
        covered = set()
        for idx in code:
            covered |= balls[idx]
        if len(covered) < num_words:
            # Fallback: return the greedy solution without removals
            # Rebuild
            uncovered = set(range(num_words))
            code2 = []
            while uncovered and time.time() < deadline - 2:
                best_idx = -1
                best_count = 0
                for i in range(num_words):
                    cnt = len(balls[i] & uncovered)
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                if best_idx == -1:
                    break
                code2.append(best_idx)
                uncovered -= balls[best_idx]
            result = [all_words[i].tolist() for i in code2]
    
    return result
