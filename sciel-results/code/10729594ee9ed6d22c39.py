<<<<<<< SEARCH
    # Fine search
    result = minimize(side_for_angle, theta0, bounds=[(theta0 - math.pi/30, theta0 + math.pi/30)], method='bounded')
    return result.fun
=======
    # Fine search
    result = minimize_scalar(side_for_angle, bounds=(theta0 - math.pi/30, theta0 + math.pi/30), method='bounded')
    return result.fun
>>>>>>> REPLACE
