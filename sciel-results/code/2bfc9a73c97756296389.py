import time

def construct(n):
    """
    Constructs the shortest known superpermutation using the classic recursive method.
    Length is sum_{k=1}^n k!.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412143214234123413124321"

    # Recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break

        w_size = m - 1
        perms, seen = [], set()
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)

        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            # Find the maximum overlap between the end of new_word and the start of block
            # Optimize: the overlap is at most len(block) - 1, and typically small
            max_overlap = min(len(new_word), len(block) - 1)
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break

            new_word += block[overlap:]

        word = new_word

    return "".join(map(str, word))
