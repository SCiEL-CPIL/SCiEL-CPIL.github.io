import random
import time
import sys
from collections import deque

N = 12

def construct():
    # Construct the known optimal Černý automaton for N=12
    # a: cycle
    # b: fixes 0 and N-1, maps 1 to 0, and shifts others down
    trans_a = [(i + 1) % N for i in range(N)]
    trans_b = [0] * N
    trans_b[0] = 0
    trans_b[1] = 0
    for i in range(2, N - 1):
        trans_b[i] = i - 1
    trans_b[N - 1] = N - 1
    
    return [trans_a, trans_b]
