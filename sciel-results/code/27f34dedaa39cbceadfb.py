"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 75
    n = 12

    def valid(arr):
        sums = {0}
        for x in arr:
            new_sums = {s + x for s in sums}
            if new_sums & sums:
                return False
            sums = sums | new_sums
        return True

    arr = [1 << i for i in range(n)]

    for _ in range(200):
        if time.time() > deadline:
            break
        changed = False
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            for new_val in range(arr[i] - 1, 0, -1):
                if new_val in arr:
                    continue
                test = arr[:]
                test[i] = new_val
                test.sort()
                if valid(test):
                    arr = test
                    changed = True
                    break
        if not changed:
            break

    return arr
# EVOLVE-BLOCK-END
