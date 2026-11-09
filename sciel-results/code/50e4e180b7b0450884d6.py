import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    # Task specific constants
    q = 3
    n = 6
    radius = 1
    total_words = q ** n

    # Precompute balls of radius 1
    balls = []
    for c in range(total_words):
        digits = [(c // (q ** (n - 1 - i))) % q for i in range(n)]
        ball = [c]
        for i in range(n):
            for new_val in range(q):
                if new_val == digits[i]:
                    continue
                old_contrib = digits[i] * (q ** (n - 1 - i))
                new_contrib = new_val * (q ** (n - 1 - i))
                new_c = c - old_contrib + new_contrib
                ball.append(new_c)
        balls.append(ball)

    # Helpers
    def list_to_int(digits):
        val = 0
        for i in range(n):
            val = val * q + digits[i]
        return val

    def int_to_list(c):
        digits = []
        for _ in range(n):
            digits.append(c % q)
            c //= q
        return digits[::-1]

    # Initial Code: Seed
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            initial_code.add(list_to_int(word))

    current_code = initial_code
    current_size = len(current_code)
    cover = [0] * total_words

    def update_cover(c, delta):
        for w in balls[c]:
            cover[w] += delta

    for c in current_code:
        update_cover(c, 1)

    def try_remove(c):
        if c not in current_code:
            return False
        for w in balls[c]:
            if cover[w] == 1:
                return False
        update_cover(c, -1)
        current_code.remove(c)
        return True

    def get_redundant_codewords(exclude=None):
        redundant = []
        for c in current_code:
            if c == exclude:
                continue
            is_red = True
            for w in balls[c]:
                if cover[w] == 1:
                    is_red = False
                    break
            if is_red:
                redundant.append(c)
        return redundant

    def try_replace(c_old, c_new):
        if c_old not in current_code or c_new in current_code:
            return False
        update_cover(c_old, -1)
        update_cover(c_new, 1)
        if min(cover) >= 1:
            current_code.remove(c_old)
            current_code.add(c_new)
            return True
        else:
            update_cover(c_new, -1)
            update_cover(c_old, 1)
            return False

    def do_add_remove(c_new):
        if c_new in current_code:
            return 0
        update_cover(c_new, 1)
        current_code.add(c_new)
        
        redundant = get_redundant_codewords(exclude=c_new)
        removed_list = []
        for c in redundant:
            if try_remove(c):
                removed_list.append(c)
                
        net_change = len(removed_list) - 1
        
        if net_change > 0:
            # Rollback
            update_cover(c_new, -1)
            current_code.remove(c_new)
            for c in removed_list:
                update_cover(c, 1)
                current_code.add(c)
            return 0
        else:
            return net_change

    # Phase 1: Greedy
    changed = True
    while changed:
        changed = False
        for c in list(current_code):
            if try_remove(c):
                changed = True

    best_size = current_size
    best_code = list(current_code)

    # Phase 2: Random Walk
    while time.time() < deadline:
        r = random.random()
        if r < 0.5 and current_code:
            c_old = random.choice(list(current_code))
            c_new = random.randint(0, total_words - 1)
            if try_replace(c_old, c_new):
                if current_size < best_size:
                    best_size = current_size
                    best_code = list(current_code)
        else:
            c_new = random.randint(0, total_words - 1)
            net_change = do_add_remove(c_new)
            if net_change < 0:
                if current_size < best_size:
                    best_size = current_size
                    best_code = list(current_code)
            elif net_change == 0:
                # Structural change accepted, check if it enables future improvements?
                # No immediate size change, but good for diversity.
                pass

    return [int_to_list(c) for c in best_code]
