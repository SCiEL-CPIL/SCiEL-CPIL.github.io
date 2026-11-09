"""XOR-circuit synthesis via Common Subexpression Elimination."""
import time
from collections import Counter

# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = 85.0
    
    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE
    
    # Parse matrix
    try:
        import numpy as np
        if isinstance(matrix, np.ndarray):
            m, n = matrix.shape
            rows = []
            for i in range(m):
                val = 0
                for j in range(n):
                    if matrix[i, j] != 0:
                        val |= (1 << j)
                rows.append(val)
    except Exception:
        m = len(matrix)
        n = len(matrix[0]) if m > 0 else 0
        rows = []
        for i in range(m):
            val = 0
            for j in range(n):
                if matrix[i][j] != 0:
                    val |= (1 << j)
            rows.append(val)
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Wire numbering:
    # 0..n-1: input wires
    # n: zero wire
    # n+1, n+2, ...: gate output wires
    
    # State:
    # wire_val: dict wire -> bitmask value
    wire_val = {}
    for i in range(n):
        wire_val[i] = (1 << i)
    wire_val[n] = 0
    
    # gates: list of (a, b) where gate produces wire n+1+idx
    gates = []
    
    # needed: Counter of bitmasks we still need to produce as outputs
    needed = Counter(rows)
    
    # Remove expressions already available
    for expr in list(needed.keys()):
        if expr in wire_val.values():
            del needed[expr]
    
    # CSE loop: find pairs (a,b) of available wires where a^b is needed with count>=2
    while needed and elapsed_ok():
        best = None  # (a, b, e, count)
        for e, cnt in needed.items():
            if cnt < 2:
                continue
            for a, va in wire_val.items():
                b_val = va ^ e
                for b, vb in wire_val.items():
                    if vb == b_val:
                        if best is None or cnt > best[3]:
                            best = (a, b, e, cnt)
                        break
        
        if best is None:
            # No CSE benefit; compute one needed expression directly
            e = list(needed.keys())[0]
            del needed[e]
            # Decompose e
            done = False
            for a, va in wire_val.items():
                b_val = va ^ e
                for b, vb in wire_val.items():
                    if vb == b_val:
                        gates.append((a, b))
                        w = n + 1 + len(gates) - 1
                        wire_val[w] = e
                        done = True
                        break
                if done:
                    break
            if not done:
                # Fallback: build from bits
                bits = [j for j in range(n) if (e >> j) & 1]
                if len(bits) <= 1:
                    continue
                cur = bits[0]
                for j in bits[1:]:
                    gates.append((cur, j))
                    cur = n + 1 + len(gates) - 1
                    wire_val[cur] = wire_val.get(cur, 0)
                wire_val[cur] = e
        else:
            a, b, e, cnt = best
            gates.append((a, b))
            w = n + 1 + len(gates) - 1
            wire_val[w] = e
            del needed[e]
    
    # Compute remaining needed expressions
    for e in list(needed.keys()):
        if not elapsed_ok():
            break
        del needed[e]
        done = False
        for a, va in wire_val.items():
            b_val = va ^ e
            for b, vb in wire_val.items():
                if vb == b_val:
                    gates.append((a, b))
                    w = n + 1 + len(gates) - 1
                    wire_val[w] = e
                    done = True
                    break
            if done:
                break
        if not done:
            bits = [j for j in range(n) if (e >> j) & 1]
            if len(bits) <= 1:
                continue
            cur = bits[0]
            for j in bits[1:]:
                gates.append((cur, j))
                cur = n + 1 + len(gates) - 1
                wire_val[cur] = wire_val.get(cur, 0)
            wire_val[cur] = e
    
    # Determine output wires
    outputs = []
    for row in rows:
        out = None
        for w, v in wire_val.items():
            if v == row:
                out = w
                break
        if out is None:
            out = n  # fallback to zero
        outputs.append(out)
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
