import itertools
import time
import random

def construct(q, n, radius):
    deadline = time.time() + 140  # safety margin before 150s limit
    
    def hamming_dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])
    
    # Generate all words
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    all_set = set(tuple(w) for w in all_words)
    
    # Precompute balls of radius 1 for each word
    balls = {}
    for w in all_words:
        ball = {tuple(w)}
        for i in range(n):
            for v in range(q):
                if v != w[i]:
                    nw = w[:]
                    nw[i] = v
                    ball.add(tuple(nw))
        balls[tuple(w)] = ball
    
    # Greedy set cover
    uncovered = set(all_set)
    code = []
    
    while uncovered and time.time() < deadline - 30:
        # Find word whose ball covers most uncovered
        best_word = None
        best_count = -1
        for w in all_words:
            covered = len(balls[tuple(w)] & uncovered)
            if covered > best_count:
                best_count = covered
                best_word = w
        if best_count <= 0:
            break
        code.append(best_word)
        uncovered -= balls[tuple(best_word)]
    
    # Local search: try to remove codewords
    def is_covered(code_list):
        if not code_list:
            return False
        covered = set()
        for c in code_list:
            covered |= balls[tuple(c)]
        return covered == all_set
    
    # Try removing each codeword
    changed = True
    while changed and time.time() < deadline - 20:
        changed = False
        for i in range(len(code)):
            trial = code[:i] + code[i+1:]
            if is_covered(trial):
                code = trial
                changed = True
                break
    
    # Try swaps: replace code[i] with some other word
    if time.time() < deadline - 10:
        for i in range(len(code)):
            if time.time() > deadline - 10:
                break
            trial = code[:]
            trial[i] = None  # placeholder
            # Try replacing with a word that covers more
            # Actually, just try random swaps
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                j = random.randint(0, len(code) - 1)
                new_word = random.choice(all_words)
                if new_word in code and new_word != code[j]:
                    continue
                trial = code[:]
                trial[j] = new_word
                if is_covered(trial):
                    code = trial
    
    return code
