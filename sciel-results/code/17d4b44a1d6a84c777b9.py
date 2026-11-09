import time
from fractions import Fraction

def construct():
    start_time = time.time()
    limit_time = 50.0  # Safety margin
    
    # We know k=7 is impossible (max sum < 1)
    # We start from k=8
    
    def dfs(remaining, start_denom, terms_left, current):
        if remaining == 0:
            return current[:]
        if terms_left == 0:
            return None
        if remaining < 0:
            return None
            
        # Pruning: 
        # The maximum sum we can achieve with 'terms_left' terms starting from 'start_denom'
        # is sum(1/i for i in range(start_denom, start_denom + terms_left))
        # If this max sum < remaining, we can't reach it.
        # This calculation is expensive if done every time, but terms_left is small.
        max_possible = sum(Fraction(1, i) for i in range(start_denom, start_denom + terms_left))
        if max_possible < remaining:
            return None
        
        # The next denominator d must be >= start_denom
        # Also 1/d <= remaining => d >= 1/remaining
        # And 1/d + max_sum(terms_left-1, d+1) >= remaining
        
        min_d = max(start_denom, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # Upper bound for d:
        # We need to leave enough sum for the remaining terms.
        # The minimum sum for the remaining 'terms_left - 1' terms is effectively 0 (if denominators go to infinity),
        # but practically, 1/d must be <= remaining.
        # Also, if terms_left == 1, d must be exactly 1/remaining.
        
        if terms_left == 1:
            if remaining.numerator == 1:
                d = remaining.denominator
                if d >= min_d and d <= 10000000:
                    return current + [d]
            return None

        # Iterate possible denominators
        # Upper bound for d:
        # 1/d + (terms_left - 1)/d > remaining ? No.
        # 1/d must be small enough so that remaining - 1/d can be formed by terms_left-1 terms starting from d+1.
        # A simple upper bound is d such that 1/d <= remaining.
        # Also, we know that if d is too large, the remaining sum is too large for the remaining terms.
        # Specifically, remaining - 1/d <= sum(1/i for i in range(d+1, d+1+terms_left-1))
        # This is a tight bound. Let's just iterate d from min_d upwards.
        # We can stop if 1/d < remaining / terms_left ? No, that's for average.
        
        # Let's just iterate. The space is small.
        # d_max is roughly remaining.denominator / remaining.numerator
        # But we can prune if remaining - 1/d is too big for the rest.
        
        # Simple bound: d < remaining.denominator / remaining.numerator
        # Actually, if d >= 1/remaining, then 1/d <= remaining.
        
        # Let's estimate max_d.
        # If we pick d, the remaining sum is R' = remaining - 1/d.
        # We need R' <= sum_{j=1}^{terms_left-1} 1/(d+j)
        # This is approximately (terms_left-1)/d.
        # So remaining - 1/d <= (terms_left-1)/d  => remaining <= terms_left/d => d <= terms_left / remaining.
        
        max_d = min(10000000, (terms_left * remaining.denominator) // remaining.numerator + 1)
        
        for d in range(min_d, max_d + 1):
            if time.time() - start_time > limit_time:
                return "TIMEOUT"
                
            next_remaining = remaining - Fraction(1, d)
            if next_remaining < 0:
                break # Since d is increasing, 1/d is decreasing, so next_remaining will only increase? 
                # Wait. If d increases, 1/d decreases. So remaining - 1/d INCREASES.
                # So if next_remaining < 0, it means 1/d > remaining.
                # As d increases, 1/d gets smaller, so next_remaining gets bigger.
                # So if next_remaining < 0, we should continue? 
                # No, if 1/d > remaining, then d < 1/remaining.
                # But we started d from min_d which is ceil(1/remaining).
                # So for d >= min_d, 1/d <= remaining. So next_remaining >= 0.
                # So this break condition is wrong. 
                # If next_remaining < 0, it means we subtracted too much? 
                # Ah, if d is small, 1/d is large. 
                # min_d ensures 1/d <= remaining. So next_remaining >= 0 always.
            
            result = dfs(next_remaining, d + 1, terms_left - 1, current + [d])
            if result is not None:
                return result
        
        return None

    for k in range(8, 257):
        res = dfs(Fraction(1, 1), 5, k, [])
        if res is not None and res != "TIMEOUT":
            # Verify
            if sum(Fraction(1, x) for x in res) == 1 and all(5 <= x <= 10000000 for x in res) and len(set(res)) == len(res):
                return res
        elif res == "TIMEOUT":
            break

    # Fallback if search fails or times out (should not happen for k=8)
    # A valid solution with k=11: 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12+1/13+1/14+1/15 is not 1.
    # Just return a hardcoded solution if necessary.
    # 1 = 1/2 + 1/3 + 1/6. 
    # Split 1/2 = 1/3 + 1/6. 
    # We need denominators >= 5.
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60 + 1/61 + 1/62 + 1/63 + 1/64 + 1/65 + 1/66 + 1/67 + 1/68 + 1/69 + 1/70 + 1/71 + 1/72 + 1/73 + 1/74 + 1/75 + 1/76 + 1/77 + 1/78 + 1/79 + 1/80 + 1/81 + 1/82 + 1/83 + 1/84 + 1/85 + 1/86 + 1/87 + 1/88 + 1/89 + 1/90 + 1/91 + 1/92 + 1/93 + 1/94 + 1/95 + 1/96 + 1/97 + 1/98 + 1/99 + 1/100 + 1/101 + 1/102 + 1/103 + 1/104 + 1/105 + 1/106 + 1/107 + 1/108 + 1/109 + 1/110 + 1/111 + 1/112 + 1/113 + 1/114 + 1/115 + 1/116 + 1/117 + 1/118 + 1/119 + 1/120 + 1/121 + 1/122 + 1/123 + 1/124 + 1/125 + 1/126 + 1/127 + 1/128 + 1/129 + 1/130 + 1/131 + 1/132 + 1/133 + 1/134 + 1/135 + 1/136 + 1/137 + 1/138 + 1/139 + 1/140 + 1/141 + 1/142 + 1/143 + 1/144 + 1/145 + 1/146 + 1/147 + 1/148 + 1/149 + 1/150 + 1/151 + 1/152 + 1/153 + 1/154 + 1/155 + 1/156 + 1/157 + 1/158 + 1/159 + 1/160 + 1/161 + 1/162 + 1/163 + 1/164 + 1/165 + 1/166 + 1/167 + 1/168 + 1/169 + 1/170 + 1/171 + 1/172 + 1/173 + 1/174 + 1/175 + 1/176 + 1/177 + 1/178 + 1/179 + 1/180 + 1/181 + 1/182 + 1/183 + 1/184 + 1/185 + 1/186 + 1/187 + 1/188 + 1/189 + 1/190 + 1/191 + 1/192 + 1/193 + 1/194 + 1/195 + 1/196 + 1/197 + 1/198 + 1/199 + 1/200 + 1/201 + 1/202 + 1/203 + 1/204 + 1/205 + 1/206 + 1/207 + 1/208 + 1/209 + 1/210 + 1/211 + 1/212 + 1/213 + 1/214 + 1/215 + 1/216 + 1/217 + 1/218 + 1/219 + 1/220 + 1/221 + 1/222 + 1/223 + 1/224 + 1/225 + 1/226 + 1/227 + 1/228 + 1/229 + 1/230 + 1/231 + 1/232 + 1/233 + 1/234 + 1/235 + 1/236 + 1/237 + 1/238 + 1/239 + 1/240 + 1/241 + 1/242 + 1/243 + 1/244 + 1/245 + 1/246 + 1/247 + 1/248 + 1/249 + 1/250 + 1/251 + 1/252 + 1/253 + 1/254 + 1/255 + 1/256
    # This is just a range, sum is not 1.
    # I will rely on the DFS finding a solution for k=8.
    raise RuntimeError("No solution found")
