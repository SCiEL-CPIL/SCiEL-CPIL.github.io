from fractions import Fraction

def construct():
    # Known minimal solution with 8 distinct denominators >= 5 summing to 1:
    # 1/5 + 1/6 + 1/7 + 1/9 + 1/10 + 1/15 + 1/35 + 1/45 = 1
    # Verify:
    # LCM of 5,6,7,9,10,15,35,45 = 3150
    # 3150/5 = 630
    # 3150/6 = 525
    # 3150/7 = 450
    # 3150/9 = 350
    # 3150/10 = 315
    # 3150/15 = 210
    # 3150/35 = 90
    # 3150/45 = 70
    # Sum = 630+525+450+350+315+210+90+70 = 2640... wait that's not 3150
    # Let me recompute: 630+525=1155, +450=1605, +350=1955, +315=2270, +210=2480, +90=2570, +70=2640
    # 2640 != 3150. So this is NOT a valid solution.
    
    # Let me find a valid 8-term solution.
    # Actually, let me use a different known approach.
    # 1 = 1/2 + 1/3 + 1/6 but we need denominators >= 5.
    
    # Let me try to construct one systematically.
    # Start with 1/5. Remaining = 4/5.
    # 4/5 = 1/6 + 19/30. 19/30... 
    # 4/5 = 1/6 + 1/7 + 23/105. 
    # 23/105 = 1/10 + 47/210.
    # 47/210 = 1/15 + 12/105 = 1/15 + 4/35.
    # 4/35 = 1/35 + 3/35 = 1/35 + 1/12 + 1/14... no that's getting messy.
    
    # Let me just use a known valid solution.
    # 1/5 + 1/6 + 1/7 + 1/9 + 1/10 + 1/15 + 1/35 + 1/45
    # LCM = 3150
    # 630 + 525 + 450 + 350 + 315 + 210 + 90 + 70 = 2640. Not 3150.
    
    # Try: 1/5 + 1/6 + 1/7 + 1/9 + 1/10 + 1/12 + 1/35 + 1/45
    # LCM of 5,6,7,9,10,12,35,45 = 12600
    # 12600/5=2520, /6=2100, /7=1800, /9=1400, /10=1260, /12=1050, /35=360, /45=280
    # Sum = 2520+2100+1800+1400+1260+1050+360+280 = 10770. Not 12600.
    
    # Let me use a computational search with strong pruning to find a valid 8-term solution.
    # Or better yet, let me just verify a known solution from literature.
    
    # Actually, the minimum number of terms for Egyptian fractions with all denominators > 4 
    # (i.e., >= 5) summing to 1 is known. Let me just do a quick DFS with good pruning
    # to find one, but do it efficiently.
    
    def find_solution(k):
        def dfs(current_num, current_den, next_d, rem, path):
            # current_num/current_den is the sum so far, as a reduced fraction
            if rem == 0:
                if current_num == current_den:
                    return path
                return None
            
            deficit_num = current_den - current_num
            deficit_den = current_den
            if deficit_num <= 0:
                return None
            
            # Reduce deficit
            from math import gcd
            g = gcd(deficit_num, deficit_den)
            deficit_num //= g
            deficit_den //= g
            
            # min_d: 1/d <= deficit => d >= deficit_den/deficit_num
            min_d_from_deficit = (deficit_den + deficit_num - 1) // deficit_num
            min_d = max(next_d, min_d_from_deficit)
            
            # max_d: rem/d >= deficit => d <= rem * deficit_den / deficit_num
            max_d = (rem * deficit_den) // deficit_num + 1
            
            if min_d > max_d:
                return None
            
            for d in range(min_d, max_d + 1):
                # Pruning: rem/d < deficit means we can't reach the deficit
                if rem * deficit_den < d * deficit_num:
                    break
                
                # new_sum = current_num/current_den + 1/d = (current_num*d + current_den) / (current_den*d)
                new_num = current_num * d + current_den
                new_den = current_den * d
                
                # Reduce
                g = gcd(new_num, new_den)
                new_num //= g
                new_den //= g
                
                res = dfs(new_num, new_den, d + 1, rem - 1, path + [d])
                if res is not None:
                    return res
            return None
        
        return dfs(0, 1, 5, k, [])
    
    for k in range(8, 257):
        sol = find_solution(k)
        if sol:
            # Verify
            s = sum(Fraction(1, d) for d in sol)
            if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                return sol
    raise RuntimeError("No solution found")
