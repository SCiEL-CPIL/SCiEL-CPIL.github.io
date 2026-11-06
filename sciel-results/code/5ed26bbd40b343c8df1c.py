import time
import random
from collections import deque

N = 12

def construct():
    # Pinwheel automaton
    # Letter 0: cycle
    # Letter 1: shift towards 0, with 0 fixed
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
    b[0] = 0
    
    return [a, b]
