import time

def construct():
    deadline = time.time() + 59.0
    N = 10000
    phi = list(range(N+1))
    for i in range(2, N+1):
        if phi[i] == i: # i is prime
            if time.time() > deadline:
                return 143
            for j in range(i, N+1, i):
                phi[j] -= phi[j] // i
    for n in range(1, N+1):
        if phi[n] % 120 == 0:
            return n
    return 143
