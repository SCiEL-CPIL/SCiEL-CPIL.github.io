# EVOLVE-BLOCK-START
def construct():
    def phi(n):
        result = n
        p = 2
        x = n
        while p * p <= x:
            if x % p == 0:
                while x % p == 0:
                    x = x // p
                result -= result // p
            p += 1
        if x > 1:
            result -= result // x
        return result

    for n in range(1, 200):
        if phi(n) % 120 == 0:
            return n
    return 143
# EVOLVE-BLOCK-END
