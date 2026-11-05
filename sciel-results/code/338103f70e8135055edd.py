import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin

    # Consistent integer <-> word mapping
    def word_to_int(word):
        w = 0
        for d in word:
            w = w * q + d
        return w

    def int_to_word(w):
        digits = []
        for _ in range(n):
            digits.append(w % q)
            w //= q
        digits.reverse()
        return digits

    # Precompute balls (Hamming neighbors) for all words
    balls = []
    for w in range(q**n):
        word = int_to_word(w)
        neighbors = [w]
        for i in range(n):
            orig = word[i]
            for delta in range(1, q):
                new_digit = (orig + delta) % q
                new_word = word[:]
                new_word[i] = new_digit
                neighbors.append(word_to_int(new_word))
        balls.append(neighbors)

    # Initial code: Tetracode × full space on remaining coords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            word = h + list(tail)
            initial_code.append(word_to_int(word))

    best_code = initial_code[:]
    best_size = len(best_code)

    def get_cover_array(code):
        cover = np.zeros(q**n, dtype=np.int8)
        for c in code:
            for v in balls[c]:
                cover[v] += 1
        return cover

    def is_valid_cover(cover):
        return np.all(cover >= 1)

    # Greedy removal with multiple restarts and different orderings
    for restart in range(500):
        if time.time() > deadline:
            break

        code = initial_code[:]
        random.shuffle(code)

        # Apply symmetry-breaking: sort by first few digits to break some symmetry
        code.sort(key=lambda w: (int_to_word(w)[0], int_to_word(w)[1], w))

        cover = get_cover_array(code)

        removed_any = False
        for i in range(len(code)):
            if time.time() > deadline:
                break

            c = code[i]
            # Tentatively remove
            for v in balls[c]:
                cover[v] -= 1

            # Check if all still covered
            if np.all(cover >= 1):
                code.pop(i)
                removed_any = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
            else:
                # Restore
                for v in balls[c]:
                    cover[v] += 1

        if not removed_any:
            continue

        if best_size <= 73:
            break

    # Add-then-remove local search to escape local minima
    for iteration in range(200):
        if time.time() > deadline:
            break

        code = best_code[:]
        cover = get_cover_array(code)

        # Try adding random words and then removing
        improved = False
        for _ in range(50):  # Try 50 random additions per iteration
            if time.time() > deadline:
                break

            # Pick a random word to add
            rand_w = random.randint(0, q**n - 1)
            if rand_w in code:
                continue

            # Add it
            code.append(rand_w)
            for v in balls[rand_w]:
                cover[v] += 1

            # Now try to remove words greedily
            random.shuffle(code)
            for i in range(len(code)):
                if time.time() > deadline:
                    break

                c = code[i]
                for v in balls[c]:
                    cover[v] -= 1

                if np.all(cover >= 1):
                    code.pop(i)
                    improved = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                else:
                    for v in balls[c]:
                        cover[v] += 1

        if not improved:
            # Try a different strategy: add multiple words then remove
            for _ in range(10):
                if time.time() > deadline:
                    break

                code = best_code[:]
                cover = get_cover_array(code)

                # Add 2-3 random words
                num_add = random.randint(2, 3)
                added = set()
                for _ in range(num_add):
                    rand_w = random.randint(0, q**n - 1)
                    if rand_w not in code and rand_w not in added:
                        code.append(rand_w)
                        added.add(rand_w)
                        for v in balls[rand_w]:
                            cover[v] += 1

                # Remove greedily
                random.shuffle(code)
                for i in range(len(code)):
                    if time.time() > deadline:
                        break

                    c = code[i]
                    for v in balls[c]:
                        cover[v] -= 1

                    if np.all(cover >= 1):
                        code.pop(i)
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                    else:
                        for v in balls[c]:
                            cover[v] += 1

    # Final polish: try more aggressive removal with different orderings
    code = best_code[:]
    cover = get_cover_array(code)

    for attempt in range(100):
        if time.time() > deadline:
            break

        # Try different removal strategies
        strategy = attempt % 3
        if strategy == 0:
            random.shuffle(code)
        elif strategy == 1:
            code.sort(key=lambda w: int_to_word(w)[0])
        else:
            code.sort(key=lambda w: -len([v for v in balls[w] if cover[v] == 1]))

        improved = False
        for i in range(len(code)):
            if time.time() > deadline:
                break

            c = code[i]
            for v in balls[c]:
                cover[v] -= 1

            if np.all(cover >= 1):
                code.pop(i)
                improved = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
            else:
                for v in balls[c]:
                    cover[v] += 1

        if not improved:
            break

    # Convert best_code to list of lists
    result = [int_to_word(w) for w in best_code]
    return result
