"""Improved covering code construction for K_3(6,1)."""
import itertools
import time
import random

def construct(q, n, radius):
    """Construct a covering code of length n over alphabet size q with covering radius radius.
    
    For q=3, n=6, radius=1, this implements a greedy algorithm with local search
    to find a small covering code.
    """
    if q == 3 and n == 6 and radius == 1:
        return _ternary_6_1()
    
    # General fallback: direct product construction
    if n <= 1:
        return [[i] for i in range(q)]
    
    # Simple greedy for other cases
    return _greedy_cover(q, n, radius)


def _ternary_6_1():
    """Specific optimized construction for ternary length-6 radius-1 covering code."""
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Start with the tetracode-based construction (9 * 3^2 = 81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    # All possible words
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    def covered_set(code):
        """Return the set of indices covered by the code (within Hamming distance <= 1)."""
        covered = set()
        for cw in code:
            covered.add(word_to_idx[cw])
            for i in range(6):
                for v in range(3):
                    if v != cw[i]:
                        w = list(cw)
                        w[i] = v
                        idx = word_to_idx[tuple(w)]
                        covered.add(idx)
        return covered
    
    def uncovered_count(code):
        covered = covered_set(code)
        return len(all_words) - len(covered)
    
    # Greedy removal: try to remove words that don't change coverage
    code = list(initial_code)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_covered = covered_set(code)
        for i in range(len(code)):
            if time.time() > deadline:
                break
            # Try removing code[i]
            remaining = code[:i] + code[i+1:]
            if not remaining:
                continue
            # Check if removing this word still covers everything
            # Quick check: is there any word only covered by code[i]?
            covered_remaining = covered_set(remaining)
            if len(covered_remaining) == len(all_words):
                code = remaining
                improved = True
                break
    
    # Local search: try replacing words to reduce count further
    for iteration in range(20):
        if time.time() > deadline:
            break
        # Try random replacements
        best_code = list(code)
        best_score = len(code)
        
        for attempt in range(500):
            if time.time() > deadline:
                break
            trial_code = list(code)
            # Replace one random word with another random word
            idx = random.randint(0, len(trial_code) - 1)
            new_word = tuple(random.randint(0, 2) for _ in range(6))
            if new_word not in [tuple(w) for w in trial_code if w is not trial_code[idx]]:
                trial_code[idx] = list(new_word)
                # Check if still covers all
                if len(covered_set(trial_code)) == len(all_words):
                    # Try to remove another word
                    for j in range(len(trial_code)):
                        remaining = trial_code[:j] + trial_code[j+1:]
                        if remaining and len(covered_set(remaining)) == len(all_words):
                            trial_code = remaining
                            break
                    if len(trial_code) < best_score:
                        best_code = trial_code
                        best_score = len(trial_code)
        
        if best_score < len(code):
            code = best_code
    
    return code


def _greedy_cover(q, n, radius):
    """Greedy set cover for general case."""
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    def ball(word):
        """All words within Hamming distance <= radius of word."""
        result = [word]
        for i in range(n):
            for v in range(q):
                if v != word[i]:
                    w = list(word)
                    w[i] = v
                    # Only add if distance is still <= radius
                    dist = sum(1 for j in range(n) if w[j] != word[j])
                    if dist <= radius:
                        result.append(tuple(w))
        return result
    
    covered = set()
    code = []
    
    while len(covered) < len(all_words) and len(code) < 1000:
        best_word = None
        best_gain = 0
        for w in all_words:
            if w in covered:
                continue
            b = ball(w)
            gain = len([x for x in b if x not in covered])
            if gain > best_gain:
                best_gain = gain
                best_word = w
        if best_word is None:
            break
        code.append(list(best_word))
        for w in ball(best_word):
            covered.add(w)
    
    return code


if __name__ == "__main__":
    # Quick test
    code = construct(3, 6, 1)
    print(f"Found {len(code)} codewords")
