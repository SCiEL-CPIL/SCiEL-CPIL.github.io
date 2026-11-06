# after computing v and updating children_count:
# 1) store any output in fast that is now fully consumed
for x in list(fast):
    if x in outputs and children_count[x] == 0:
        ops.append(['store', x]); fast.remove(x); slow.add(x)
# 2) delete any non-output in fast that is fully consumed
for x in list(fast):
    if x not in outputs and children_count[x] == 0:
        ops.append(['delete', x]); fast.remove(x)
# 3) budget: while len(fast) > B:
#      find non-output with children_count==0 -> delete (shouldn't exist after step2)
#      else find non-output with children_count>0 -> store (spill)
#      (outputs in fast with children_count>0 are kept; if over budget and only outputs remain, infeasible)
